from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..prompters.wan_prompter import WanPrompter
import torch
import os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict

from ..vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_dit import WanModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..models.wan_lora import WanLoraWrapper
from torch.nn import functional as F

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model

MAX_AREA_FOR_RESOLUTION = {
    "720p": 720 * 1280,
    "480p": 480 * 832,
    "1080p": 1080 * 1920,
    "2k": 1440 * 2560,
    "4k": 2160 * 3840,
}


def get_target_image_size(
    raw_width,
    raw_height,
    resolution="720p",
    height_division_factor=16,
    width_division_factor=16,
):
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    if resolution not in MAX_AREA_FOR_RESOLUTION:
        logger.error(f"{resolution} is not supported")
        return None, None
    max_area = MAX_AREA_FOR_RESOLUTION[resolution]
    aspect_ratio = raw_height / raw_width
    lat_h = round(
        np.sqrt(max_area * aspect_ratio)
        // vae_stride[1]
        // patch_size[1]
        * patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio)
        // vae_stride[2]
        // patch_size[2]
        * patch_size[2]
    )
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    h = (
        (h + height_division_factor - 1)
        // height_division_factor
        * height_division_factor
    )
    w = (w + width_division_factor - 1) // width_division_factor * width_division_factor

    return w, h


@torch.amp.autocast("cuda", dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


class TeleAILongVideoV1(BasePipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        tokenizer_path=None,
        vae_dtype=torch.bfloat16,
        sample_solver="euler",  # or unipc
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.sample_solver = sample_solver
        logger.info(f"set sampler solver to {sample_solver}")
        if sample_solver == "euler":
            self.scheduler = FlowMatchScheduler(
                shift=5, sigma_min=0.0, extra_one_step=True
            )
        else:
            raise NotImplementedError(f"solver {sample_solver} is not supported")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_parallel = False
        self.batch_cfg = False
        self.vae_dtype = vae_dtype
        self.vae_parallel = False
        self.sr_pipe = None

    def load_loras(self, lora_configs):
        lora_wrapper = WanLoraWrapper(self.dit)
        for lora_config in lora_configs:
            lora_path = lora_config["path"]
            strength = lora_config.get("strength", 1.0)
            lora_name = lora_wrapper.load_lora(lora_path)
            lora_wrapper.apply_lora(lora_name, strength)
            logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

    def disable_cpu_offload(self):
        logger.info("disable cpu offload")
        self.cpu_offload_model_names = self.model_names
        self.load_models_to_device(self.model_names)
        self.cpu_offload_model_names = []

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if not self.dit.fsdp_flag:
            dtype = next(iter(self.dit.parameters())).dtype
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if not self.vae_parallel:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @classmethod
    def from_model_manager(
        cls,
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        compile=False,
        dit_fsdp=False,
        parallelism: int = 1,
        use_cfg_parallel: bool = False,
        vae_parallel: bool = False,
        vae_dtype=torch.bfloat16,
        dit_device_ids: Optional[List] = None,
        vae_device_ids: Optional[List] = None,
        sample_solver: str = "euler",
        lora_configs: List[Dict] = [],
        enable_vram_management: bool = False,
        vram_management_config: Dict = {},
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            vae_dtype=vae_dtype,
            sample_solver=sample_solver,
        )
        pipe.fetch_models(model_manager)
        if lora_configs:
            pipe.load_loras(lora_configs)
        pipe.vae_parallel = vae_parallel
        pipe.dit_parallel = parallelism > 1
        if parallelism > 1:
            if use_cfg_parallel:
                pipe.batch_cfg = True
                pipe.dit.enable_cfgp()
                cfg_degree = 2
            else:
                cfg_degree = 1
            sp_ulysses_degree = parallelism // cfg_degree
            if sp_ulysses_degree > 1:
                pipe.dit.enable_usp()
            if enable_vram_management:
                pipe.enable_vram_management(**vram_management_config)
            pipe.dit = ParallelModel(
                pipe.dit,
                cfg_degree=cfg_degree,
                sp_ulysses_degree=sp_ulysses_degree,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_fsdp
                else None,
                device="cuda",
                master_port=random.randint(39991, 65535),
                compile=compile,
                name="dit",
                device_ids=dit_device_ids,
            )
            if vae_parallel:
                vae_parallel_num = (
                    len(vae_device_ids) if vae_device_ids else parallelism
                )
                logger.info(f"set vae parallel num is {vae_parallel_num}")
                pipe.vae.set_parallelism(vae_parallel_num)
                pipe.vae = ParallelModel(
                    pipe.vae,
                    cfg_degree=1,
                    sp_ring_degree=1,
                    sp_ulysses_degree=vae_parallel_num,
                    tp_degree=1,
                    master_port=random.randint(1025, 12800),
                    device="cuda",
                    name="vae",
                    device_ids=vae_device_ids,
                )
        else:
            if compile:
                pipe.enable_compile_dit()
        return pipe

    def enable_compile_dit(self):
        logger.info("enable torch compile")
        self.dit = torch.compile(self.dit, mode="default")

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(
            prompt, positive=positive, device=self.device
        )
        return {"context": prompt_emb}

    def encode_image(
        self,
        image,
        end_image,
        num_frames,
        height,
        width,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        image = self.preprocess_image(image.resize((width, height), Image.LANCZOS)).to(
            self.device
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(
                end_image.resize((width, height), Image.LANCZOS)
            ).to(self.device)
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae.encode(
            [vae_input.to(dtype=self.vae_dtype, device=self.device)],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return y

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        latents = self.vae.encode(
            input_video.to(dtype=self.vae_dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents = latents.to(dtype=self.torch_dtype, device=self.device)
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents.to(dtype=self.vae_dtype),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def predict_noise_with_cfg(
        self,
        dit,
        latents,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        extra_input,
        attn_impl,
        cfg_scale=1.0,
        use_cfg_zero_star=False,
        freq_start_idx=0,
        freq_interval=1,
    ):
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                freq_start_idx=freq_start_idx,
                freq_interval=freq_interval,
            )
        if not self.batch_cfg:
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                freq_start_idx=freq_start_idx,
                freq_interval=freq_interval,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
                freq_start_idx=freq_start_idx,
                freq_interval=freq_interval,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                context=context,
                attn_impl=attn_impl,
                **extra_input,
                freq_start_idx=freq_start_idx,
                freq_interval=freq_interval,
            )
        if use_cfg_zero_star:
            positive_flat = noise_pred_posi.view(1, -1)
            negative_flat = noise_pred_nega.view(1, -1)
            alpha = optimized_scale(positive_flat, negative_flat)
            alpha = alpha.view(1, *([1] * (len(noise_pred_posi.shape) - 1)))
            alpha = alpha.to(noise_pred_posi.dtype)
            noise_pred = noise_pred_nega * alpha + cfg_scale * (
                noise_pred_posi - noise_pred_nega * alpha
            )
        else:
            noise_pred = noise_pred_nega + cfg_scale * (
                noise_pred_posi - noise_pred_nega
            )
        return noise_pred

    def preparer_condition_video(
        self, frames, this_frame_interval, next_frame_interval
    ):
        b, c, t, h, w = frames.shape
        ratio = this_frame_interval // next_frame_interval
        new_nframes = t * ratio

        n_chunks = max(4 // next_frame_interval, 1)
        nframes_per_chunk = (new_nframes // n_chunks + 2) // 4 * 4 + 1  # round it up

        new_frames = torch.zeros(b, c, new_nframes, h, w).to(
            frames.device, frames.dtype
        )
        masks = torch.zeros(b, new_nframes, h // 8, w // 8).to(
            frames.device, frames.dtype
        )

        for chunk_idx in range(n_chunks):
            new_frames[:, :, chunk_idx * nframes_per_chunk] = frames[
                :, :, chunk_idx * nframes_per_chunk // ratio
            ]
            masks[:, chunk_idx * nframes_per_chunk] = 1

            if chunk_idx < n_chunks - 1:
                new_frames[:, :, (chunk_idx + 1) * nframes_per_chunk - 1] = frames[
                    :, :, ((chunk_idx + 1) * nframes_per_chunk - 1) // ratio
                ]
                masks[:, (chunk_idx + 1) * nframes_per_chunk - 1] = 1
            else:
                end_idx = min(n_chunks * nframes_per_chunk - 1, new_nframes - 1)
                new_frames[:, :, end_idx] = frames[:, :, end_idx // ratio]
                masks[:, end_idx] = 1

        anchors = []
        anchors.extend(list(range(t))[:: max(4 // this_frame_interval, 1)])
        for idx in anchors:
            new_frames[:, :, idx * ratio] = frames[:, :, idx]
            masks[:, idx * ratio] = 1

        condition_frames = []  # List[b, c, t, h, w]
        condition_masks = []

        for chunk_idx in range(n_chunks):
            if chunk_idx == 0 or chunk_idx < n_chunks - 1:
                condition_frames.append(
                    new_frames[
                        :,
                        :,
                        chunk_idx * nframes_per_chunk : (chunk_idx + 1)
                        * nframes_per_chunk,
                    ]
                )
                condition_masks.append(
                    masks[
                        :,
                        chunk_idx * nframes_per_chunk : (chunk_idx + 1)
                        * nframes_per_chunk,
                    ]
                )
            else:
                # keep the same shape for batch infer
                last_chunk_frames = torch.zeros_like(condition_frames[-1])
                last_chunk_mask = torch.zeros_like(condition_masks[-1])

                frames_left = min(
                    new_nframes - chunk_idx * nframes_per_chunk, nframes_per_chunk
                )
                last_chunk_frames[:, :, :frames_left] = new_frames[
                    :,
                    :,
                    chunk_idx * nframes_per_chunk : chunk_idx * nframes_per_chunk
                    + frames_left,
                ]
                last_chunk_mask[:, :frames_left] = masks[
                    :,
                    chunk_idx * nframes_per_chunk : chunk_idx * nframes_per_chunk
                    + frames_left,
                ]

                condition_frames.append(last_chunk_frames)
                condition_masks.append(last_chunk_mask)

        return torch.concat(condition_frames).to(
            dtype=self.torch_dtype, device=self.device
        ), torch.concat(condition_masks).to(dtype=self.torch_dtype, device=self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        seed=None,
        rand_device: Optional[str] = None,
        height=480,
        width=832,
        num_frames=81,
        base_cfg_scale=5.5,
        st_cfg_scale=5.5,
        base_num_inference_steps=40,
        st_num_inference_steps=40,
        base_sigma_shift=5.0,
        st_sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        attn_impl="sdpa",
        use_cfg_zero_star: bool = False,
        zero_init_steps=-1,
        st_scale: int = 2,
        sr_scale: int = 1,
        sr_cfg_scale: float = 5.5,
        sr_num_inference_steps: int = 10,
        sr_sigma_shift: float = 5,
        sr_timestep_range: tuple = (0, 500),
    ):
        # Parameter check
        logger.info(f"start genereate {num_frames*st_scale} {width}x{height} frames")
        height = int(height // sr_scale)
        width = int(width // sr_scale)
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        # Initialize noise
        if rand_device is None:
            rand_device = self.device
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = None
        if base_cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        ref_latent = None
        if input_image is not None:
            self.load_models_to_device(["vae"])
            ref_latent = self.encode_image(
                input_image, end_image, num_frames, height, width, **tiler_kwargs
            )

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # base Denoise
        self.load_models_to_device(["dit"])

        # Scheduler
        self.scheduler.set_timesteps(
            base_num_inference_steps,
            shift=base_sigma_shift,
        )

        # denoise stage 1
        for progress_id, timestep in enumerate(
            tqdm(
                progress_bar_cmd(self.scheduler.timesteps), desc="denoise with base dit"
            )
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            if progress_id <= zero_init_steps:
                noise_pred = latents * 0
            else:
                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    noise_pred = self.predict_noise_with_cfg(
                        dit=self.dit,
                        latents=input_latent,
                        timestep=timestep,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        extra_input=extra_input,
                        attn_impl=attn_impl,
                        cfg_scale=base_cfg_scale,
                        use_cfg_zero_star=use_cfg_zero_star,
                        freq_interval=st_scale,
                    ).to(input_latent.device)

            # Scheduler
            latents = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                latents,
                denoising_strength=1.0,
            )
        if st_scale == 1 and sr_scale > 1 and self.sr_pipe is not None:
            latents = self.sr_pipe.sr_process_tensor(
                latents,
                0.333,
                prompt_emb_posi,
                prompt_emb_nega,
                False,
                sr_num_inference_steps,
                "TeleAI-SR-1.3B",
                sr_scale,
                sr_timestep_range,
                attn_impl,
                sr_cfg_scale,
                sr_sigma_shift,
                add_init_noise=True,
            )
        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        # st
        if st_scale > 1:
            frames = torch.concat(frames, dim=2).unsqueeze(0)
            frames = self.video_st_process(
                frames,
                st_scale,
                tiler_kwargs,
                st_num_inference_steps,
                st_sigma_shift,
                prompt_emb_posi,
                prompt_emb_nega,
                extra_input,
                attn_impl,
                cfg_scale=st_cfg_scale,
                use_cfg_zero_star=use_cfg_zero_star,
                seed=seed,
                rand_device=rand_device,
                sr_scale=sr_scale,
                sr_cfg_scale=sr_cfg_scale,
                sr_num_inference_steps=sr_num_inference_steps,
                sr_sigma_shift=sr_sigma_shift,
                sr_timestep_range=sr_timestep_range,
            )
        else:
            frames = self.tensor2video(frames[0])
        self.load_models_to_device([])
        return frames

    def video_st_process(
        self,
        frames,
        st_scale,
        tile_kwargs,
        st_num_inference_steps,
        st_sigma_shift,
        prompt_emb_posi,
        prompt_emb_nega,
        extra_input,
        attn_impl,
        cfg_scale,
        use_cfg_zero_star,
        seed,
        rand_device,
        sr_scale: int = 1,
        sr_cfg_scale: float = 5.5,
        sr_num_inference_steps: int = 10,
        sr_sigma_shift: float = 5,
        sr_timestep_range: tuple = (0, 500),
    ):
        condition_chunks, masks = self.preparer_condition_video(
            frames,
            st_scale,
            1,
        )
        b, c, t, h, w = condition_chunks.shape
        ref_latents = self.encode_video(condition_chunks, **tile_kwargs)
        masks = torch.concat(
            [torch.repeat_interleave(masks[:, 0:1], repeats=4, dim=1), masks[:, 1:]],
            dim=1,
        )
        masks = masks.view(b, masks.shape[1] // 4, 4, h // 8, w // 8)
        masks = masks.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
        noise = self.generate_noise(
            ref_latents.shape,
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(ref_latents.device, dtype=self.torch_dtype)
        ref_latents = torch.concat([masks, ref_latents], dim=1)
        latent_list = []
        self.load_models_to_device(["dit"])
        for i in tqdm(range(b), desc="all st denoise"):
            self.scheduler.set_timesteps(st_num_inference_steps, shift=st_sigma_shift)
            latents = noise[i : i + 1]
            b, c, f, h, w = latents.shape
            for progress_id, timestep in enumerate(
                tqdm((self.scheduler.timesteps), desc="st denoise")
            ):
                timestep = timestep.unsqueeze(0).to(
                    dtype=self.torch_dtype, device=self.device
                )
                input_latent = latents
                if ref_latents is not None:
                    input_latent = torch.cat(
                        [input_latent, ref_latents[i : i + 1]], dim=1
                    )
                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    noise_pred = self.predict_noise_with_cfg(
                        dit=self.dit,
                        latents=input_latent,
                        timestep=timestep,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        extra_input=extra_input,
                        attn_impl=attn_impl,
                        cfg_scale=cfg_scale,
                        use_cfg_zero_star=use_cfg_zero_star,
                        freq_start_idx=i * f,
                        freq_interval=1,
                    ).to(input_latent.device)

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    latents,
                    denoising_strength=1.0,
                )
            if sr_scale > 1 and self.sr_pipe is not None:
                latents = self.sr_pipe.sr_process_tensor(
                    latents,
                    0.333,
                    prompt_emb_posi,
                    prompt_emb_nega,
                    False,
                    sr_num_inference_steps,
                    "TeleAI-SR-1.3B",
                    sr_scale,
                    sr_timestep_range,
                    attn_impl,
                    sr_cfg_scale,
                    sr_sigma_shift,
                    add_init_noise=True,
                )
            latent_list.append(latents)
        self.load_models_to_device(["vae"])
        frame_list = []
        for latent in latent_list:
            frames = self.decode_video(latent, **tile_kwargs)
            frames = self.tensor2video(frames[0])
            frame_list.extend(frames)
        return frame_list

    def __del__(self):
        del self.dit
        del self.vae
