from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..prompters.wan_prompter import WanPrompter
import torch
import torch.nn.functional as F
import os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict

from ..models.wan_video_dit import WanModel
from ..models.wan_video_vae import WanVideoVAE
from ..schedulers.flow_match_v2 import FlowMatchSchedulerV2
from ..models.wan_lora import WanLoraWrapper

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model
import math


class TeleVideoSRPipeline(BasePipeline):
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
            self.scheduler = FlowMatchSchedulerV2(
                shift=5, sigma_min=0.0, extra_one_step=True
            )
        else:
            raise NotImplementedError(f"solver {sample_solver} is not supported")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
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

    def fetch_models(self, model_manager: ModelManager):
        self.dit = model_manager.fetch_model("wan_video_dit")
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.vae = model_manager.fetch_model("wan_video_vae")

    def disable_cpu_offload(self):
        logger.info("disable cpu offload")
        self.cpu_offload_model_names = self.model_names
        self.load_models_to_device(self.model_names)
        self.cpu_offload_model_names = []

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
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        attn_impl,
        cfg_scale=1.0,
    ):
        if cfg_scale == 1.0:
            return self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                attn_impl=attn_impl,
            )
        if not self.batch_cfg:
            noise_pred_posi = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                attn_impl=attn_impl,
            )
            noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                **prompt_emb_nega,
                attn_impl=attn_impl,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                context=context,
                attn_impl=attn_impl,
            )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    def upsample_embeddings(self, latent, scale=2):
        b, c, t, h, w = latent.shape  ### (1,16,21,64,96)
        latent = latent.cpu()
        resized_latent = (
            F.interpolate(  ### (1,16,21,128,192)
                latent.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w),
                size=(h * scale, w * scale),
                mode="nearest",
            )
            .reshape(b, t, c, h * scale, w * scale)
            .permute(0, 2, 1, 3, 4)
        )
        return resized_latent

    def sample_block_noise(
        self, frame=21, ch=16, height=160, width=90, eps=1e-6, gamma=-1 / 3, scale=2
    ):
        assert height % scale == 0 and width % scale == 0
        block_h = height // scale
        block_w = width // scale
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(scale**2),
            torch.eye(scale**2) * (1 + eps - gamma)
            + torch.ones(scale**2, scale**2) * gamma,
        )

        block_number = frame * ch * block_h * block_w
        noise = dist.sample((block_number,))

        return rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=frame,
            c=ch,
            h=block_h,
            w=block_w,
            p=scale,
            q=scale,
        ).permute(1, 0, 2, 3)  # c, t, h, w

    def sr_process_tensor(
        self,
        latents,
        init_sigma,
        prompt_emb_posi,
        prompt_emb_nega,
        enable_magcache,
        num_inference_steps,
        model_type,
        sr_scale,
        timestep_range,
        attn_impl,
        cfg_scale,
        sigma_shift,
        seed=42,
        rand_device="cpu",
        add_init_noise: bool = False,
    ):
        if add_init_noise:
            # add init noise
            noise = self.generate_noise(
                latents.shape,
                seed=seed,
                device=rand_device,
                dtype=torch.float32,
            )
            noise = noise.to(dtype=self.torch_dtype, device=self.device)
            latents = (1 - init_sigma) * latents + init_sigma * noise
        # Scheduler
        self.scheduler.sigma_max = timestep_range[1] / 1000
        self.scheduler.sigma_min = 0
        self.scheduler.set_timesteps(
            num_inference_steps,
            shift=sigma_shift,
        )
        # Denoise
        self.load_models_to_device(["dit"])
        if enable_magcache:
            self.dit.set_mag_cache(num_inference_steps, model_type)
        else:
            self.dit.reset_mag_cache()
        # upsample
        latents = self.upsample_embeddings(
            latents, scale=sr_scale
        )  ### ### (1,16,21,128,192)
        b, c, t, h, w = latents.shape
        # renoise
        end = timestep_range[1] / 1000
        start = init_sigma
        alpha = (1 - end) / (1 - start)
        beta = math.sqrt(end**2 - (alpha * start) ** 2)
        gamma = -((alpha * start) ** 2) / (beta**2)

        block_noise = (
            self.sample_block_noise(
                frame=t, height=h, width=w, gamma=gamma, eps=1e-5, scale=sr_scale
            )
            .unsqueeze(0)
            .to(dtype=self.torch_dtype, device="cpu")
        )
        latents = alpha * latents + beta * block_noise
        latents = latents.to(dtype=self.torch_dtype, device=self.device)
        for progress_id, timestep in enumerate(
            tqdm(self.scheduler.timesteps, desc="generate sr video")
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    attn_impl=attn_impl,
                    cfg_scale=cfg_scale,
                ).to(input_latent.device)

            # Scheduler
            latents = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                latents,
                denoising_strength=1
                / (self.scheduler.sigma_max - self.scheduler.sigma_min),
            )
        self.load_models_to_device([])
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        input_video: List = [],
        negative_prompt="",
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        model_type="",
        attn_impl="sdpa",
        enable_magcache=False,
        sr_scale: float = 2,
        timestep_range: tuple = (0, 500),
        seed=42,
        rand_device="cpu",
        init_sigma=0.333,
    ):
        # Parameter check
        num_frames = len(input_video)
        if num_frames < 1:
            logger.warning("input video is empty")
            return None
        if num_frames % 4 != 1:
            num_frames = num_frames // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )
        width, height = input_video[0].size
        height, width = self.check_resize_height_width(height, width)
        logger.info(
            f"start genereate {num_frames} {width}x{height} with sr scale {sr_scale}"
        )

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        self.load_models_to_device(["vae"])
        input_video = self.preprocess_images(input_video)
        input_video = torch.stack(input_video, dim=2).to(
            dtype=self.torch_dtype, device=self.device
        )
        print("input video: ", input_video.shape)
        input_video = (
            torch.nn.functional.interpolate(
                input_video[0].permute(1, 0, 2, 3),
                size=(height, width),
                mode="bilinear",
            )
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )
        print("input video: ", input_video.shape)
        latents = self.encode_video(input_video, **tiler_kwargs).to(
            dtype=self.torch_dtype, device=self.device
        )
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = None
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        # Decode
        latents = self.sr_process_tensor(
            latents,
            init_sigma,
            prompt_emb_posi,
            prompt_emb_nega,
            enable_magcache,
            num_inference_steps,
            model_type,
            sr_scale,
            timestep_range,
            attn_impl,
            cfg_scale,
            sigma_shift,
            seed=seed,
            rand_device=rand_device,
            add_init_noise=True,
        )
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        return frames

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def __del__(self):
        del self.dit
        del self.vae
