from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.models.sparse_attention import SparseAttentionParams
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
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
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from ..schedulers.unipc import FlowUniPCMultistepScheduler
from ..schedulers.lcm import LCMScheduler
from ..models.wan_lora import WanLoraWrapper

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model

# sparse attn
from functools import lru_cache

try:
    from tele_fuser.radial_attn.attn import MaskMap
    from tele_fuser.radial_attn.utils import find_nearest_size_fast_with_constraints
except ImportError:
    print("failed to import radial_attn, please install TeleFuser")


MAX_AREA_FOR_RESOLUTION = {
    "720p": 720 * 1280,
    "480p": 480 * 832,
    "1080p": 1080 * 1920,
    "2k": 1440 * 2560,
    "4k": 2160 * 3840,
}


@lru_cache(maxsize=32)
def prepare_sparse_attention(width=1280, height=768, num_frames=81, resolution="720p"):
    if resolution not in MAX_AREA_FOR_RESOLUTION:
        logger.error(f"resolution {resolution} is not supported")
        return None, None
    max_area = MAX_AREA_FOR_RESOLUTION[resolution]
    new_size = find_nearest_size_fast_with_constraints(
        width, height, num_frames, max_area
    )
    if new_size is None:
        logger.error("failed to find right size for sparse attn")
        return (
            None,
            None,
        )
    width, height = new_size
    print(f"generate widht {width}, {height}")
    temperal_downsample = [True, True, False]
    patch_size = (1, 2, 2)
    upsampling_factor = 8
    vae_scale_factor_temporal = 2 ** sum(temperal_downsample)
    num_frames = 1 + num_frames // (vae_scale_factor_temporal * patch_size[0])
    mod_value = upsampling_factor * patch_size[1]
    frame_size = int(height // mod_value) * int(width // mod_value)
    logger.info("Sparse Video token configuration:")
    logger.info(f"- Temporal scale factor: {vae_scale_factor_temporal}")
    logger.info(f"- Number of frames: {num_frames}")
    logger.info(f"- Frame size: {frame_size}")
    logger.info(f"- Total tokens: {frame_size * num_frames}")
    return MaskMap(video_token_num=frame_size * num_frames, num_frame=num_frames), (
        width,
        height,
    )


SUPPORTED_ASPECT_RATIO_LIST = {
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "1:1",
    "2:3",
    "3:2",
}


def get_target_video_size_from_ratio(
    aspect_ratio, resolution="720p", height_division_factor=16, width_division_factor=16
):
    if aspect_ratio not in SUPPORTED_ASPECT_RATIO_LIST:
        logger.error(f"aspect ratio {aspect_ratio} is not supported")
    w, h = aspect_ratio.split(":")
    return get_target_image_size(
        int(w), int(h), resolution, height_division_factor, width_division_factor
    )


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


class WanVideoPipeline(BasePipeline):
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
        elif sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
        elif sample_solver == "lcm":
            self.scheduler = LCMScheduler(num_train_timesteps=1000, shift=5)
        else:
            raise NotImplementedError(f"solver {sample_solver} is not supported")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae", "image_encoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_parallel = False
        self.batch_cfg = False
        self.vae_dtype = vae_dtype
        self.vae_parallel = False

    def load_loras(self, lora_configs):
        lora_wrapper = WanLoraWrapper(self.dit)
        for lora_config in lora_configs:
            lora_path = lora_config["path"]
            strength = lora_config.get("strength", 1.0)
            lora_name = lora_wrapper.load_lora(lora_path)
            lora_wrapper.apply_lora(lora_name, strength)
            logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

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
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

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
                master_port=random.randint(12800, 65535),
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
                pipe.dit = torch.compile(pipe.dit, mode="default")
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
        clip_context = self.image_encoder.encode_image([image])
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
            # if self.dit.has_image_pos_emb:
            clip_context = torch.concat(
                [clip_context, self.image_encoder.encode_image([end_image])], dim=1
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
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return clip_context, y

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
        latents,
        cn_images,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        clip_feature,
        extra_input,
        attn_impl,
        cfg_scale=1.0,
        use_cfg_zero_star=False,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        if cfg_scale == 1.0:
            return self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                clip_feature=clip_feature,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                sparse_params=sparse_params,
            )
        if not self.batch_cfg:
            noise_pred_posi = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                clip_feature=clip_feature,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                sparse_params=sparse_params,
            )
            noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                clip_feature=clip_feature,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
                sparse_params=sparse_params,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            if clip_feature is not None:
                clip_feature = torch.cat([clip_feature, clip_feature], dim=0)

            if cn_images:
                cn_images = torch.cat([cn_images, cn_images], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                context=context,
                clip_feature=clip_feature,
                attn_impl=attn_impl,
                **extra_input,
                sparse_params=sparse_params,
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

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device: Optional[str] = None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        cfg_skip_ratio=0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=0,
        model_type="",
        progress_bar_cmd=tqdm,
        cn_images=None,
        attn_impl="sdpa",
        enable_riflex=False,
        enable_magcache=False,
        zero_init_steps=-1,
        sparse_params: Optional[SparseAttentionParams] = None,
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

        # Scheduler
        if self.sample_solver == "euler":
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
            )
        elif self.sample_solver == "unipc":
            self.scheduler.set_timesteps(
                num_inference_steps, device=self.device, shift=sigma_shift
            )
        elif self.sample_solver == "lcm":
            self.scheduler.set_timesteps(
                num_inference_steps, shift=sigma_shift, device=self.device, seed=seed
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

        if cn_images is not None:
            self.load_models_to_device(["vae"])
            if isinstance(cn_images, tuple):  # 将cn_images 分别处理
                stacked_cn_images = []
                for per_cn_images in cn_images:
                    per_cn_images = self.preprocess_images(per_cn_images)
                    per_cn_images = torch.stack(per_cn_images, dim=2).to(
                        dtype=self.torch_dtype, device=self.device
                    )
                    per_cn_images = self.encode_video(per_cn_images, **tiler_kwargs).to(
                        dtype=self.torch_dtype, device=self.device
                    )
                    stacked_cn_images.append(per_cn_images)
                stacked_cn_images = torch.cat(stacked_cn_images, dim=1)
                cn_images = stacked_cn_images

            else:
                cn_images = self.preprocess_images(cn_images)
                cn_images = torch.stack(cn_images, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )
                cn_images = self.encode_video(cn_images, **tiler_kwargs).to(
                    dtype=self.torch_dtype, device=self.device
                )

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = None
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        clip_context = None
        ref_latent = None
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            clip_context, ref_latent = self.encode_image(
                input_image, end_image, num_frames, height, width, **tiler_kwargs
            )

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        if enable_riflex:
            self.dit.enable_riflex(k=6, L_test=noise.shape[2])
        else:
            self.dit.disable_riflex()
        if enable_magcache:
            self.dit.set_mag_cache(num_inference_steps, model_type)
        else:
            self.dit.reset_mag_cache()
            if tea_cache_l1_thresh > 0:
                self.dit.set_tea_cache(
                    num_inference_steps, tea_cache_l1_thresh, model_type
                )
            else:
                self.dit.reset_tea_cache()

        for progress_id, timestep in enumerate(
            tqdm(progress_bar_cmd(self.scheduler.timesteps), desc="generate video")
        ):
            # sparse update
            if sparse_params is not None:
                sparse_params.update(numeral_timestep=progress_id)

            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            if cn_images is not None:
                input_latent = torch.cat([input_latent, cn_images], dim=1)

            if (
                progress_id > (1 - cfg_skip_ratio) * num_inference_steps
                and not self.batch_cfg
            ):
                cfg_scale = 1.0
            if progress_id <= zero_init_steps:
                noise_pred = latents * 0
            else:
                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    noise_pred = self.predict_noise_with_cfg(
                        latents=input_latent,
                        cn_images=cn_images,
                        timestep=timestep,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        clip_feature=clip_context,
                        extra_input=extra_input,
                        attn_impl=attn_impl,
                        cfg_scale=cfg_scale,
                        use_cfg_zero_star=zero_init_steps >= 0,
                        sparse_params=sparse_params,
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
        return frames

    def __del__(self):
        del self.dit
        del self.vae
