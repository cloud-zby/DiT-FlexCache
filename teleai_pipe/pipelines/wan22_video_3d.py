from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from ..prompters.wan_prompter import WanPrompter
import torch
import os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict, Union
import torch.nn as nn
from einops import repeat, reduce

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
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from ..schedulers.unipc import FlowUniPCMultistepScheduler
from ..models.wan_video_dit import RMSNorm

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model

from ..models.wan_lora import WanLoraWrapper
from math import ceil


def context_scheduler(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


class WanVideo22S2VPipeline_long(BasePipeline):
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
        self.num_train_timesteps = 1000
        logger.info(f"set sampler solver to {sample_solver}")
        if sample_solver in ["euler", "multi_euler"]:
            self.scheduler = FlowMatchScheduler(
                shift=5, sigma_min=0.0, extra_one_step=True
            )
        elif sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        else:
            raise NotImplementedError(f"solver {sample_solver} is not supported")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit_high: WanModel = None
        self.dit_low: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = [
            "text_encoder",
            "dit_high",
            "dit_low",
            "vae",
            "image_encoder",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_parallel = False
        self.batch_cfg = False
        self.vae_dtype = vae_dtype
        self.vae_parallel = False
        self.pose_embedding = None

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
        if not self.dit_high.fsdp_flag:
            logger.warning("dit high with vram management")
            dtype = next(iter(self.dit_high.parameters())).dtype
            enable_vram_management(
                self.dit_high,
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
        if not self.dit_low.fsdp_flag:
            logger.warning("dit low with vram management")
            dtype = next(iter(self.dit_low.parameters())).dtype
            enable_vram_management(
                self.dit_low,
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
                    computation_dtype=self.vae_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def load_loras(self, lora_configs):
        for dit_name in ["dit_high", "dit_low"]:
            dit_lora_configs = lora_configs.get(dit_name, {})
            dit = getattr(self, dit_name)
            lora_wrapper = WanLoraWrapper(dit)
            for lora_config in dit_lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_path} with strength: {strength}")

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
        self.dit_high, self.dit_low = model_manager.fetch_model(
            "wan_video_dit", index=2
        )
        self.vae = model_manager.fetch_model("wan_video_vae")
        #######################add pose
        concat_dim = 4
        self.pose_embedding = nn.Sequential(
            nn.Conv3d(
                3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        state_dict_new = {}
        if model_manager.state_dict_new_module_high:
            # print("model_manager.state_dict_new_module_high:",model_manager.state_dict_new_module_high)
            for key in model_manager.state_dict_new_module_high:
                if "pose_embedding_high" in key:
                    # print("========> pose_embedding_high OK！")
                    state_dict_new[key.split("pose_embedding_high.")[1]] = (
                        model_manager.state_dict_new_module_high[key]
                    )
                else:
                    pass
                    # print("========> pose_embedding_high key NO！")
            self.pose_embedding.load_state_dict(state_dict_new, strict=True)
            self.pose_embedding.to(dtype=self.torch_dtype, device=self.device)
        if model_manager.state_dict_new_module_high:
            original_state_dict = self.dit_high.state_dict()
            updated_state_dict = {}
            updated_state_dict = {
                key: model_manager.state_dict_new_module_high[(key)]
                for key in original_state_dict.keys()
                if (key) in model_manager.state_dict_new_module_high
            }
            original_state_dict.update(updated_state_dict)
            self.dit_high.load_state_dict(original_state_dict)

        state_dict_new = {}
        for key in model_manager.state_dict_new_module:
            if "randomref_embedding_pose" in key:
                state_dict_new[key.split("randomref_embedding_pose.")[1]] = (
                    model_manager.state_dict_new_module[key]
                )

        # update self attentions parameters.
        if model_manager.state_dict_self_attn:
            original_state_dict = self.dit_high.state_dict()
            updated_state_dict = {
                key: model_manager.state_dict_self_attn[("pipe.dit." + key)]
                for key in original_state_dict.keys()
                if ("pipe.dit." + key) in model_manager.state_dict_self_attn
            }

            original_state_dict.update(updated_state_dict)
            self.dit_high.load_state_dict(original_state_dict)

    @classmethod
    def from_model_manager(
        cls,
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        parallelism: int = 1,
        dit_high_compile=False,
        dit_low_compile=False,
        dit_high_fsdp=False,
        dit_low_fsdp=False,
        dit_high_cfg_parallel: bool = False,
        dit_low_cfg_parallel: bool = False,
        vae_parallel: bool = False,
        vae_dtype=torch.bfloat16,
        dit_device_ids: Optional[List] = None,
        vae_device_ids: Optional[List] = None,
        sample_solver: str = "unipc",
        enable_vram_management: bool = False,
        vram_management_config: Dict = {},
        lora_configs: Dict = {},
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
        pipe.fetch_models(model_manager)  ###
        if lora_configs:
            pipe.load_loras(lora_configs)
        pipe.vae_parallel = vae_parallel
        pipe.dit_parallel = parallelism > 1
        pipe.dit_high.fsdp_flag = dit_high_fsdp
        pipe.dit_low.fsdp_flag = dit_low_fsdp
        if enable_vram_management:
            pipe.enable_vram_management(**vram_management_config)
        if parallelism > 1:
            pipe.batch_cfg = True
            if dit_high_cfg_parallel:
                cfg_degree = 2
                pipe.dit_high.enable_cfgp()
                sp_ulysses_degree = parallelism // cfg_degree
            else:
                cfg_degree = 1
                sp_ulysses_degree = parallelism
            if sp_ulysses_degree > 1:
                pipe.dit_high.enable_usp()
            pipe.dit_high = ParallelModel(
                pipe.dit_high,
                cfg_degree=cfg_degree,
                sp_ulysses_degree=sp_ulysses_degree,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_high_fsdp
                else None,
                device="cuda",
                master_port=random.randint(12800, 33800),
                compile=dit_high_compile,
                name="dit_high",
                device_ids=dit_device_ids,
            )
            if dit_low_cfg_parallel:
                cfg_degree = 2
                pipe.dit_low.enable_cfgp()
                sp_ulysses_degree = parallelism // cfg_degree
            else:
                cfg_degree = 1
                sp_ulysses_degree = parallelism
            if sp_ulysses_degree > 1:
                pipe.dit_low.enable_usp()
            pipe.dit_low = ParallelModel(
                pipe.dit_low,
                cfg_degree=cfg_degree,
                sp_ulysses_degree=sp_ulysses_degree,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_low_fsdp
                else None,
                device="cuda",
                master_port=random.randint(33800, 65535),
                compile=dit_low_compile,
                name="dit_low",
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
        self.dit_high = torch.compile(self.dit_high, mode="default")
        self.dit_low = torch.compile(self.dit_low, mode="default")

    def denoising_model(self):
        return self.dit_high, self.dit_low

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
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(
                self.device
            )
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
        cn_latents,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        extra_input,
        attn_impl,
        cfg_scale=1.0,
    ):
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
        if not self.batch_cfg:
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=False,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = dit.forward(
                x=latents,
                cn_latents=cn_latents,
                timestep=timestep,
                context=context,
                attn_impl=attn_impl,
                **extra_input,
            )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(
            image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {})
        )
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a list of PIL.Image to torch.Tensor
        video = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        control_video: Optional[list[Image.Image]] = None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device: Optional[str] = None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale_high=5.0,
        cfg_scale_low=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        model_type="",
        progress_bar_cmd=tqdm,
        attn_impl="sdpa",
        enable_magcache_dit_high=False,
        enable_magcache_dit_low=False,
        boundary: float = 0.875,
        context_size: int = 21,
        context_overlap: int = 4,
        infer_final=True,
    ):
        logger.info(f"start generate {num_frames} {width}x{height} frames")

        # --- 参数检查 ---
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # --- Scheduler ---
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # --- 初始化 latent ---
        rand_device = rand_device or self.device
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        # --- 编码 prompt ---
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # --- 编码 reference image ---
        ref_latent = None
        if input_image is not None:
            self.load_models_to_device(["vae"])
            ref_latent = self.encode_image(
                input_image, end_image, num_frames, height, width
            )

        # --- 编码 control_data_list ---
        cn_latents_list = []
        first_feature_per_seg = []
        if self.pose_embedding is not None and control_video is not None:
            self.pose_embedding.to(self.device).to(torch.bfloat16)
            control_data = self.preprocess_video(control_video)
            control_data = control_data[:, :, :num_frames, :, :]
            # print("control_data shape",control_data.shape)

            # 滑动窗口 context_scheduler
            context_queue = list(
                context_scheduler(
                    step=0,
                    num_steps=31,
                    num_frames=noise.shape[2],
                    context_size=context_size,
                    context_stride=1,
                    context_overlap=context_overlap,
                )
            )
            context_step = min(
                1, int(np.ceil(np.log2(noise.shape[2] / context_size))) + 1
            )
            context_queue[-1] = [
                e % noise.shape[2]
                for e in range(
                    noise.shape[2] - context_size * context_step,
                    noise.shape[2],
                    context_step,
                )
            ]
            global_context = context_queue

            # 生成 control_data_list + first_feature_per_seg
            for ii in global_context:
                control_data_per = control_data[:, :, ii[0] * 4 : ii[-1] * 4 + 1, :, :]
                control_data_list.append(
                    self.pose_embedding(
                        (
                            torch.cat(
                                [
                                    control_data_per[:, :, :1].repeat(1, 1, 3, 1, 1),
                                    control_data_per,
                                ],
                                dim=2,
                            )
                            / 255.0
                        ).to(self.device)
                    ).to(torch.bfloat16)
                )
                first_feature_per_seg.append(latents[:, :, ii[0] : ii[0] + 2].clone())

        # --- 额外输入 ---
        extra_input = self.prepare_extra_input(latents)

        # --- DIT 设置 ---
        self.load_models_to_device(["dit_high"])
        if enable_magcache_dit_high:
            self.dit_high.set_mag_cache(num_inference_steps, model_type)
        else:
            self.dit_high.reset_mag_cache()

        dit = self.dit_high
        cfg_scale = cfg_scale_high

        # --- 长视频滑动窗口去噪 ---
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps, desc="generate video")
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            noise_pred_out = torch.zeros_like(latents)
            counter = torch.zeros_like(latents[:, :1])

            # DIT 切换
            if timestep < boundary * self.num_train_timesteps and dit is self.dit_high:
                self.load_models_to_device(["dit_low"])
                dit = self.dit_low
                cn_latents_list = None
                cfg_scale = cfg_scale_low
                if enable_magcache_dit_low:
                    self.dit_low.set_mag_cache(
                        num_inference_steps, model_type, progress_id
                    )
                else:
                    self.dit_low.reset_mag_cache()

            # 遍历 segment
            for i_index, context in enumerate(cn_latents_list or [None]):
                if cn_latents_list is not None:
                    latent_model_input = latents[:, :, context, :, :]
                    condition = cn_latents_list[i_index][:, :, : len(context), :, :]
                    condition = rearrange(
                        condition, "b c f h w -> b (f h w) c"
                    ).contiguous()
                    latent_model_input[:, :, :2, :, :] = first_feature_per_seg[i_index]
                else:
                    latent_model_input = latents
                    condition = None

                input_latent = latent_model_input
                if ref_latent is not None:
                    input_latent = torch.cat([input_latent, ref_latent], dim=1)

                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    noise_pred = self.predict_noise_with_cfg(
                        dit,
                        latents=input_latent,
                        cn_latents=condition,
                        timestep=timestep,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        extra_input=extra_input,
                        attn_impl=attn_impl,
                        cfg_scale=cfg_scale,
                    ).to(input_latent.device)

                # --- 段内帧累积 ---
                if cn_latents_list is not None:
                    for idx, frame_idx in enumerate(context):
                        if idx < 2:  # 前两帧用 first_feature_per_seg
                            continue
                        noise_pred_out[:, :, frame_idx, :, :] += noise_pred[
                            :, :, idx, :, :
                        ]
                        counter[:, :, frame_idx, :, :] += 1
                    first_feature_per_seg[i_index] = latent_model_input[
                        :, :, :2, :, :
                    ].clone()
                else:
                    noise_pred_out += noise_pred
                    counter += 1

            # --- 平均 overlapping latent ---
            latents = noise_pred_out / counter

        # --- Decode ---
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        if infer_final:
            self.resume_all()

        return frames

    def __del__(self):
        del self.dit_high
        del self.dit_low
        del self.vae


#################################################################################
class WanVideo22S2VPipeline(BasePipeline):
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
        self.num_train_timesteps = 1000
        logger.info(f"set sampler solver to {sample_solver}")
        if sample_solver in ["euler", "multi_euler"]:
            self.scheduler = FlowMatchScheduler(
                shift=5, sigma_min=0.0, extra_one_step=True
            )
        elif sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        else:
            raise NotImplementedError(f"solver {sample_solver} is not supported")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit_high: WanModel = None
        self.dit_low: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = [
            "text_encoder",
            "dit_high",
            "dit_low",
            "vae",
            "image_encoder",
        ]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_parallel = False
        self.batch_cfg = False
        self.vae_dtype = vae_dtype
        self.vae_parallel = False
        self.pose_embedding = None

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
        if not self.dit_high.fsdp_flag:
            logger.warning("dit high with vram management")
            dtype = next(iter(self.dit_high.parameters())).dtype
            enable_vram_management(
                self.dit_high,
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
        if not self.dit_low.fsdp_flag:
            logger.warning("dit low with vram management")
            dtype = next(iter(self.dit_low.parameters())).dtype
            enable_vram_management(
                self.dit_low,
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
                    computation_dtype=self.vae_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def load_loras(self, lora_configs):
        for dit_name in ["dit_high", "dit_low"]:
            dit_lora_configs = lora_configs.get(dit_name, {})
            dit = getattr(self, dit_name)
            lora_wrapper = WanLoraWrapper(dit)
            for lora_config in dit_lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_path} with strength: {strength}")

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
        self.dit_high, self.dit_low = model_manager.fetch_model(
            "wan_video_dit", index=2
        )
        self.vae = model_manager.fetch_model("wan_video_vae")
        #######################add pose
        concat_dim = 4
        self.pose_embedding = nn.Sequential(
            nn.Conv3d(
                3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        state_dict_new = {}
        if model_manager.state_dict_new_module_high:
            # print("model_manager.state_dict_new_module_high:",model_manager.state_dict_new_module_high)
            for key in model_manager.state_dict_new_module_high:
                if "pose_embedding_high" in key:
                    # print("========> pose_embedding_high OK！")
                    state_dict_new[key.split("pose_embedding_high.")[1]] = (
                        model_manager.state_dict_new_module_high[key]
                    )
                else:
                    pass
                    # print("========> pose_embedding_high key NO！")
            self.pose_embedding.load_state_dict(state_dict_new, strict=True)
            self.pose_embedding.to(dtype=self.torch_dtype, device=self.device)
        if model_manager.state_dict_new_module_high:
            original_state_dict = self.dit_high.state_dict()
            updated_state_dict = {}
            updated_state_dict = {
                key: model_manager.state_dict_new_module_high[(key)]
                for key in original_state_dict.keys()
                if (key) in model_manager.state_dict_new_module_high
            }
            original_state_dict.update(updated_state_dict)
            self.dit_high.load_state_dict(original_state_dict)

        state_dict_new = {}
        for key in model_manager.state_dict_new_module:
            if "randomref_embedding_pose" in key:
                state_dict_new[key.split("randomref_embedding_pose.")[1]] = (
                    model_manager.state_dict_new_module[key]
                )

        # update self attentions parameters.
        if model_manager.state_dict_self_attn:
            original_state_dict = self.dit_high.state_dict()
            updated_state_dict = {
                key: model_manager.state_dict_self_attn[("pipe.dit." + key)]
                for key in original_state_dict.keys()
                if ("pipe.dit." + key) in model_manager.state_dict_self_attn
            }

            original_state_dict.update(updated_state_dict)
            self.dit_high.load_state_dict(original_state_dict)

    @classmethod
    def from_model_manager(
        cls,
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        parallelism: int = 1,
        dit_high_compile=False,
        dit_low_compile=False,
        dit_high_fsdp=False,
        dit_low_fsdp=False,
        dit_high_cfg_parallel: bool = False,
        dit_low_cfg_parallel: bool = False,
        vae_parallel: bool = False,
        vae_dtype=torch.bfloat16,
        dit_device_ids: Optional[List] = None,
        vae_device_ids: Optional[List] = None,
        sample_solver: str = "unipc",
        enable_vram_management: bool = False,
        vram_management_config: Dict = {},
        lora_configs: Dict = {},
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
        pipe.fetch_models(model_manager)  ###
        if lora_configs:
            pipe.load_loras(lora_configs)
        pipe.vae_parallel = vae_parallel
        pipe.dit_parallel = parallelism > 1
        pipe.dit_high.fsdp_flag = dit_high_fsdp
        pipe.dit_low.fsdp_flag = dit_low_fsdp
        if enable_vram_management:
            pipe.enable_vram_management(**vram_management_config)
        if parallelism > 1:
            pipe.batch_cfg = True
            if dit_high_cfg_parallel:
                cfg_degree = 2
                pipe.dit_high.enable_cfgp()
                sp_ulysses_degree = parallelism // cfg_degree
            else:
                cfg_degree = 1
                sp_ulysses_degree = parallelism
            if sp_ulysses_degree > 1:
                pipe.dit_high.enable_usp()
            pipe.dit_high = ParallelModel(
                pipe.dit_high,
                cfg_degree=cfg_degree,
                sp_ulysses_degree=sp_ulysses_degree,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_high_fsdp
                else None,
                device="cuda",
                master_port=random.randint(12800, 33800),
                compile=dit_high_compile,
                name="dit_high",
                device_ids=dit_device_ids,
            )
            if dit_low_cfg_parallel:
                cfg_degree = 2
                pipe.dit_low.enable_cfgp()
                sp_ulysses_degree = parallelism // cfg_degree
            else:
                cfg_degree = 1
                sp_ulysses_degree = parallelism
            if sp_ulysses_degree > 1:
                pipe.dit_low.enable_usp()
            pipe.dit_low = ParallelModel(
                pipe.dit_low,
                cfg_degree=cfg_degree,
                sp_ulysses_degree=sp_ulysses_degree,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_low_fsdp
                else None,
                device="cuda",
                master_port=random.randint(33800, 65535),
                compile=dit_low_compile,
                name="dit_low",
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
        self.dit_high = torch.compile(self.dit_high, mode="default")
        self.dit_low = torch.compile(self.dit_low, mode="default")

    def denoising_model(self):
        return self.dit_high, self.dit_low

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
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(
                self.device
            )
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
        cn_latents,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        extra_input,
        attn_impl,
        cfg_scale=1.0,
    ):
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
        if not self.batch_cfg:
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                cn_latents=cn_latents,
                cond_flag=False,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = dit.forward(
                x=latents,
                cn_latents=cn_latents,
                timestep=timestep,
                context=context,
                attn_impl=attn_impl,
                **extra_input,
            )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(
            image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {})
        )
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        # Transform a list of PIL.Image to torch.Tensor
        video = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    # @torch.no_grad()
    # def __call__(
    #     self,
    #     prompt,
    #     negative_prompt="",
    #     input_image=None,
    #     end_image=None,
    #     control_videos: Optional[list[list[Image.Image]]] = None,  # 支持多段
    #     input_video=None,
    #     denoising_strength=1.0,
    #     seed=None,
    #     rand_device: Optional[str] = None,
    #     height=480,
    #     width=832,
    #     num_frames=81,
    #     cfg_scale_high=5.0,
    #     cfg_scale_low=5.0,
    #     num_inference_steps=50,
    #     sigma_shift=5.0,
    #     tiled=True,
    #     tile_size=(30, 52),
    #     tile_stride=(15, 26),
    #     model_type="",
    #     progress_bar_cmd=tqdm,
    #     attn_impl="sdpa",
    #     enable_magcache_dit_high=False,
    #     enable_magcache_dit_low=False,
    #     boundary: float = 0.875,
    #     overlap: int = 0,           # 新增：每段 overlap
    #     infer_final: bool = True    # 新增：最后一段是否 resume
    # ):
    #     """
    #     支持多段 control_videos 的长视频推理
    #     """
    #     all_results = []
    #     cur_image = input_image

    #     for seg_idx, seg_control in enumerate(control_videos):
    #         seg_final = (seg_idx == len(control_videos) - 1)
    #         logger.info(f"==== Pipeline 内推理第 {seg_idx+1}/{len(control_videos)} 段 ====")

    #         # ========== 保留你原来的单段逻辑 ==========
    #         height, width = self.check_resize_height_width(height, width)
    #         if num_frames % 4 != 1:
    #             num_frames = (num_frames + 2) // 4 * 4 + 1
    #             print(
    #                 f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
    #             )

    #         tiler_kwargs = {
    #             "tiled": tiled,
    #             "tile_size": tile_size,
    #             "tile_stride": tile_stride,
    #         }

    #         if self.sample_solver == "euler":
    #             self.scheduler.set_timesteps(
    #                 num_inference_steps,
    #                 denoising_strength=denoising_strength,
    #                 shift=sigma_shift,
    #             )
    #         elif self.sample_solver == "multi_euler":
    #             self.scheduler.set_stage_timesteps(
    #                 num_inference_steps[0],
    #                 num_inference_steps[1],
    #                 boundary,
    #                 shift_1=sigma_shift[0],
    #                 shift_2=sigma_shift[1],
    #                 denoising_strength=denoising_strength,
    #             )
    #         elif self.sample_solver == "unipc":
    #             self.scheduler.set_timesteps(
    #                 num_inference_steps,
    #                 device=self.device,
    #                 shift=sigma_shift,
    #             )

    #         if rand_device is None:
    #             rand_device = self.device
    #         noise = self.generate_noise(
    #             (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
    #             seed=seed,
    #             device=rand_device,
    #             dtype=torch.float32,
    #         )
    #         noise = noise.to(dtype=self.torch_dtype, device=self.device)

    #         latents = noise
    #         # 文本编码
    #         self.load_models_to_device(["text_encoder"])
    #         prompt_emb_posi = self.encode_prompt(prompt, positive=True)
    #         prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

    #         # ref image
    #         ref_latent = None
    #         if cur_image is not None:
    #             self.load_models_to_device(["vae"])
    #             ref_latent = self.encode_image(
    #                 cur_image, end_image, num_frames, height, width
    #             )

    #         # control video
    #         if self.pose_embedding is not None:
    #             self.pose_embedding.to(self.device).to(torch.bfloat16)
    #         cn_latents = None
    #         if seg_control is not None:
    #             cn_tensor = self.preprocess_video(seg_control)
    #             cn_tensor = cn_tensor[:, :, :num_frames, :, :]
    #             cn_latents = self.pose_embedding(
    #                 (torch.cat([cn_tensor[:, :, :1].repeat(1, 1, 3, 1, 1), cn_tensor], dim=2)).to(self.device).to(torch.bfloat16)
    #             )

    #         extra_input = self.prepare_extra_input(latents)

    #         # 高分辨率部分
    #         self.load_models_to_device(["dit_high"])
    #         if enable_magcache_dit_high:
    #             self.dit_high.set_mag_cache(
    #                 num_inference_steps[0]
    #                 if isinstance(num_inference_steps, tuple)
    #                 else num_inference_steps,
    #                 model_type,
    #             )
    #         else:
    #             self.dit_high.reset_mag_cache()

    #         dit = self.dit_high
    #         cfg_scale = cfg_scale_high
    #         for progress_id, timestep in enumerate(
    #             tqdm(progress_bar_cmd(self.scheduler.timesteps), desc="generate video")
    #         ):
    #             if timestep < boundary * self.num_train_timesteps and dit is self.dit_high:
    #                 self.load_models_to_device(["dit_low"])
    #                 dit = self.dit_low
    #                 cn_latents = None
    #                 cfg_scale = cfg_scale_low
    #                 if enable_magcache_dit_low:
    #                     self.dit_low.set_mag_cache(
    #                         num_inference_steps, model_type, progress_id
    #                     )
    #                 else:
    #                     self.dit_low.reset_mag_cache()
    #             timestep = timestep.unsqueeze(0).to(
    #                 dtype=self.torch_dtype, device=self.device
    #             )
    #             input_latent = latents
    #             if ref_latent is not None:
    #                 input_latent = torch.cat([input_latent, ref_latent], dim=1)

    #             ws = self.dit_high.world_size
    #             if cn_latents is not None:
    #                 b = cn_latents.size(0)
    #                 if b < ws:
    #                     cn_latents = torch.cat([cn_latents] * ws, dim=0)[:ws]

    #             with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
    #                 noise_pred = self.predict_noise_with_cfg(
    #                     dit,
    #                     latents=input_latent,
    #                     cn_latents=cn_latents,
    #                     timestep=timestep,
    #                     prompt_emb_posi=prompt_emb_posi,
    #                     prompt_emb_nega=prompt_emb_nega,
    #                     extra_input=extra_input,
    #                     attn_impl=attn_impl,
    #                     cfg_scale=cfg_scale,
    #                 ).to(input_latent.device)

    #             latents = self.scheduler.step(
    #                 noise_pred, self.scheduler.timesteps[progress_id], latents
    #             )

    #         # decode
    #         self.load_models_to_device(["vae"])
    #         frames = self.decode_video(latents, **tiler_kwargs)
    #         self.load_models_to_device([])
    #         frames = self.tensor2video(frames[0])

    #         # overlap 去重
    #         if seg_idx > 0 and overlap > 0:
    #             frames = frames[overlap:]

    #         all_results.extend(frames)

    #         # 更新 cur_image
    #         if overlap == 0:
    #             cur_image = frames[-1]
    #         else:
    #             cur_image = frames[-overlap]

    #         if seg_final and infer_final:
    #             self.resume_all()

    #     return all_results
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        control_video: Optional[list[Image.Image]] = None,  ###
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device: Optional[str] = None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale_high=5.0,
        cfg_scale_low=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        model_type="",
        progress_bar_cmd=tqdm,
        attn_impl="sdpa",
        enable_magcache_dit_high=False,
        enable_magcache_dit_low=False,
        boundary: float = 0.875,
        infer_final=True,
    ):
        logger.info(f"start genereate {num_frames} {width}x{height} frames")
        # Parameter check
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

        if self.sample_solver == "euler":
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
            )
        elif self.sample_solver == "multi_euler":
            self.scheduler.set_stage_timesteps(
                num_inference_steps[0],
                num_inference_steps[1],
                boundary,
                shift_1=sigma_shift[0],
                shift_2=sigma_shift[1],
                denoising_strength=denoising_strength,
            )
        elif self.sample_solver == "unipc":
            self.scheduler.set_timesteps(
                num_inference_steps,
                device=self.device,
                shift=sigma_shift,
            )

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
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        # Encode image
        ref_latent = None
        if input_image is not None:
            self.load_models_to_device(["vae"])
            ref_latent = self.encode_image(
                input_image, end_image, num_frames, height, width
            )
        # Encode pose
        if self.pose_embedding is not None:
            self.pose_embedding.to(self.device).to(torch.bfloat16)
        if control_video is not None:
            cn_tensor = self.preprocess_video(control_video)
            cn_tensor = cn_tensor[:, :, :num_frames, :, :]
            cn_latents = self.pose_embedding(
                (
                    torch.cat(
                        [cn_tensor[:, :, :1].repeat(1, 1, 3, 1, 1), cn_tensor], dim=2
                    )
                )
                .to(self.device)
                .to(torch.bfloat16)
            )

        # cn_latents = None
        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit_high"])
        if enable_magcache_dit_high:
            self.dit_high.set_mag_cache(
                num_inference_steps[0]
                if isinstance(num_inference_steps, tuple)
                else num_inference_steps,
                model_type,
            )
        else:
            self.dit_high.reset_mag_cache()

        dit = self.dit_high
        cfg_scale = cfg_scale_high
        for progress_id, timestep in enumerate(
            tqdm(progress_bar_cmd(self.scheduler.timesteps), desc="generate video")
        ):
            if timestep < boundary * self.num_train_timesteps and dit is self.dit_high:
                self.load_models_to_device(["dit_low"])
                dit = self.dit_low
                cn_latents = None
                cfg_scale = cfg_scale_low
                if enable_magcache_dit_low:
                    self.dit_low.set_mag_cache(
                        num_inference_steps, model_type, progress_id
                    )
                else:
                    self.dit_low.reset_mag_cache()
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)

            ws = self.dit_high.world_size  # 或者 self.world_size / dit.world_size 等
            if cn_latents is not None:
                b = cn_latents.size(0)
                if b < ws:
                    # 直接在 batch 维拼接，复制到 world_size
                    cn_latents = torch.cat([cn_latents] * ws, dim=0)[:ws]
                    # print(
                    #     f"[INFO] broadcast cn_latents from batch {b} "
                    #     f"-> {cn_latents.size(0)} to match world_size={ws}",
                    #     flush=True,
                    # )
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    dit,
                    latents=input_latent,
                    cn_latents=cn_latents,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    extra_input=extra_input,
                    attn_impl=attn_impl,
                    cfg_scale=cfg_scale,
                ).to(input_latent.device)

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents
            )
        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        # if infer_final:
        #     self.resume_all()

        return frames

    def __del__(self):
        del self.dit_high
        del self.dit_low
        del self.vae
