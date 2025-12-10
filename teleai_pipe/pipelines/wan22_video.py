from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from ..prompters.wan_prompter import WanPrompter
import torch
import os
import gc
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
from ..models.wan_video_dit import RMSNorm
from .teleai_video_sr import TeleVideoSRPipeline

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model

from ..models.wan_lora import WanLoraWrapper
from teleai_pipe.utils.profiler import ProfilingContext4Debug

from teleai_pipe.cache.metrics import CacheMetrics
from teleai_pipe.cache.log import cache_logger
from teleai_pipe.cache import FlexCache

class WanVideo22Pipeline(BasePipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        tokenizer_path=None,
        vae_dtype=torch.bfloat16,
        sample_solver="euler",  # or unipc
        dit_quantized=False,
        enable_cache=True,
        cache_dir: Optional[str] = None,
        kv_cache_config_path: Optional[str] = None,
        prefer_legacy: Optional[bool] = None,
        key_steps: Optional[List[int]] = None,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.dit_quantized = dit_quantized
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
        self.sr_pipe = None
        self.enable_cache = enable_cache
        # cache
        if enable_cache:
            

            if prefer_legacy is None:
                env_val = os.environ.get("TELEAI_PREFER_LEGACY")
                if isinstance(env_val, str) and env_val != "":
                    prefer_legacy = env_val.lower() in {"1", "true", "yes", "on"}

            if key_steps is None:
                ks_env = os.environ.get("TELEAI_KEY_STEPS")
                if isinstance(ks_env, str) and ks_env.strip():
                    try:
                        key_steps = [int(x.strip()) for x in ks_env.split(",") if x.strip()]
                    except Exception:
                        key_steps = None

            effective_cache_dir = (
                cache_dir
                if cache_dir is not None and str(cache_dir).strip() != ""
                else os.environ.get("TELEAIPIPE_CACHE_DIR", "./flexcache")
            )
            effective_kv_cfg = (
                kv_cache_config_path
                if kv_cache_config_path is not None and str(kv_cache_config_path).strip() != ""
                else os.environ.get("FLEXCACHE_KV_CONFIG")
            )

            self.cache = FlexCache(
                cache_dir=effective_cache_dir,
                device=device,
                max_cache_size_gb=10,
                kv_cache_config_path=effective_kv_cfg,
                prefer_legacy=prefer_legacy,
                key_steps=key_steps,
            )
        else:
            self.cache = None
        self.cache_metrics = CacheMetrics.get()

    def video_to_numpy_frames(self, frames):
        """将生成的视频帧转换为numpy数组格式，用于分割"""
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        return frames

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
        if not self.dit_high.fsdp_flag and not self.dit_high.async_swap_flag:
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
        if not self.dit_low.fsdp_flag and not self.dit_low.async_swap_flag:
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
            "wan_video_dit" if not self.dit_quantized else "wan_video_quantized_dit",
            index=2,
        )
        self.vae = model_manager.fetch_model("wan_video_vae")

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
        enable_dit_async_swap: bool = False,
        dit_async_swap_config: Dict = {},
        dit_quantized: bool = False,
        kv_cache_config_path=None,
        prefer_legacy: Optional[bool] = None,
        key_steps: Optional[List[int]] = None,
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
            dit_quantized=dit_quantized,
            kv_cache_config_path=kv_cache_config_path,
            prefer_legacy=prefer_legacy,
            key_steps=key_steps,
        )
        pipe.fetch_models(model_manager)
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
        if enable_dit_async_swap:
            logger.info("enable dit async swap")
            pipe.dit_low.async_swap_flag = True
            pipe.dit_high.async_swap_flag = True
            pipe.dit_low.enable_async_swap(dit_async_swap_config)
            pipe.dit_high.enable_async_swap(dit_async_swap_config)
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

    @ProfilingContext4Debug("predict noise")
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
    ):
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
        if not self.batch_cfg:
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
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
                timestep=timestep,
                context=context,
                attn_impl=attn_impl,
                **extra_input,
            )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
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
        sr_scale: int = 1,
        sr_cfg_scale: float = 5.5,
        sr_num_inference_steps: int = 10,
        sr_sigma_shift: float = 5,
        sr_timestep_range: tuple = (0, 500),
        enable_sr_magcache: bool = False,
    ):
        logger.info(f"start genereate {num_frames} {width}x{height} frames")
        height = int(height // sr_scale)
        width = int(width // sr_scale)
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

        # cache lookup 查找缓存命中
        cache_hit = False
        start_step = 0
        cache_type = "none"
        if self.cache is not None:
            cache_logger.debug("cache_reuse.lookup_start")
            cache_result = self.cache.lookup(prompt)
            if cache_result["hit"]:
                cache_hit = True
                start_step = cache_result["skip_step"]
                cache_type = cache_result["cache_type"]
                latents = cache_result["latent_state"].to(
                    device=self.device, dtype=self.torch_dtype
                )
                cache_logger.debug(
                    f"cache_reuse.hit type={cache_type} skip_step={start_step} similarity={cache_result['similarity']:.3f}"
                )
                cache_logger.debug(
                    f"cache_reuse.cached_prompt={cache_result['cached_prompt']}"
                )
                self.cache_metrics.record_lookup(True, start_step, cache_type)
            else:
                self.cache_metrics.record_lookup(False)
                cache_logger.warning("cache_reuse.lookup_miss")

        # 若未命中，正常初始化
        if not cache_hit:
            # 原有噪声初始化代码
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

        if cache_hit:
            timesteps_to_use = self.scheduler.timesteps[start_step:]
        else:
            timesteps_to_use = self.scheduler.timesteps

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        with ProfilingContext4Debug("encode text"):
            prompt_emb_posi = self.encode_prompt(prompt, positive=True)
            prompt_emb_nega = None
            if cfg_scale_high != 1.0:
                prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        # Encode image
        ref_latent = None
        if input_image is not None:
            self.load_models_to_device(["vae"])
            with ProfilingContext4Debug("encode image"):
                ref_latent = self.encode_image(
                    input_image, end_image, num_frames, height, width, **tiler_kwargs
                )

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit_high"])
        if enable_magcache_dit_high:
            self.dit_high.set_mag_cache(
                (
                    num_inference_steps[0]
                    if isinstance(num_inference_steps, tuple)
                    else num_inference_steps
                ),
                model_type,
            )
        else:
            self.dit_high.reset_mag_cache()

        dit = self.dit_high
        cfg_scale = cfg_scale_high

        latent_states_for_cache = {}
        for progress_id, timestep in enumerate(
            tqdm(progress_bar_cmd(timesteps_to_use), desc="generate video")
        ):
            if timestep < boundary * self.num_train_timesteps and dit is self.dit_high:
                self.load_models_to_device(["dit_low"])
                dit = self.dit_low
                cfg_scale = cfg_scale_low
                if enable_magcache_dit_low:
                    self.dit_low.set_mag_cache(
                        num_inference_steps, model_type, progress_id
                    )
                else:
                    self.dit_low.reset_mag_cache()
            actual_step = progress_id + start_step - 1
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    dit,
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    extra_input=extra_input,
                    attn_impl=attn_impl,
                    cfg_scale=cfg_scale,
                ).to(input_latent.device)

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, timesteps_to_use[progress_id], latents
            )

            # save key steps latent states
            current_step = actual_step + 1  
            if self.cache is not None and current_step in self.cache.key_steps:
                if not cache_hit or (cache_hit and current_step > start_step):
                    latent_states_for_cache[current_step] = latents.clone()

        # sr process
        if sr_scale > 1 and self.sr_pipe is not None:
            latents = self.sr_pipe.sr_process_tensor(
                latents,
                0.333,
                prompt_emb_posi,
                prompt_emb_nega,
                enable_sr_magcache,
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
        with ProfilingContext4Debug("decode video"):
            frames = self.decode_video(latents, **tiler_kwargs)
            frames_numpy = self.video_to_numpy_frames(frames[0])
            # save cache 
            if self.cache is not None and (not cache_hit or start_step < 25):
                if latent_states_for_cache:
                    self.cache.save(
                        prompt,
                        latent_states_for_cache,
                        self.scheduler.timesteps,
                        num_frames,
                        generated_video_frames=frames_numpy,  
                    )
                    self.cache_metrics.record_save(len(latent_states_for_cache))
                    if cache_hit:
                        cache_logger.debug(
                            f"cache_reuse.update previous_type={cache_type} previous_step={start_step}"
                        )
                    else:
                        cache_logger.debug("cache_reuse.save_new")
                    cache_logger.debug(
                        f"cache_reuse.saved_steps={list(latent_states_for_cache.keys())}"
                    )

        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        stats = self.cache_metrics.snapshot()
        cache_logger.debug(f"cache_reuse.stats {stats}")

        return frames

    def __del__(self):
        del self.dit_high
        del self.dit_low
        del self.vae
        if self.sr_pipe is not None:
            del self.sr_pipe
