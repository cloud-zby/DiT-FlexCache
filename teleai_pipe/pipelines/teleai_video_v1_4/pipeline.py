from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.models.wan_video_text_encoder import WanTextEncoder
from teleai_pipe.models.wan_video_vae import WanVideoVAE
from teleai_pipe.models.wan_video_image_encoder import WanImageEncoder
from teleai_pipe.schedulers.flow_match import FlowMatchScheduler
from teleai_pipe.prompters.wan_prompter import WanPrompter
import torch
import os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger

from teleai_pipe.vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
)
from teleai_pipe.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from teleai_pipe.models.wan_video_dit import RMSNorm
from teleai_pipe.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from teleai_pipe.models.wan_video_dit import WanModel
from teleai_pipe.models.wan_video_text_encoder import WanTextEncoder
from teleai_pipe.models.wan_video_vae import WanVideoVAE
from teleai_pipe.models.wan_video_image_encoder import WanImageEncoder
from teleai_pipe.schedulers.flow_match import FlowMatchScheduler
from teleai_pipe.models.wan_video_dit import RMSNorm
from .models.camera_pose import CAMERA_MOVEMENT_TYPE_LIST, process_pose_file

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model
from typing import Optional


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae", "image_encoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_fsdp = False

    def enable_vram_management(self, num_persistent_param_in_dit=None):
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
        if not self.dit_fsdp:
            dtype = next(iter(self.dit.parameters())).dtype
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
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
            )
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

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        compile=False,
        dit_fsdp=False,
        parallelism: int = 1,
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if parallelism > 1:
            pipe.dit.dit_fsdp = dit_fsdp
            pipe.dit.enable_usp()
            pipe.dit = ParallelModel(
                pipe.dit,
                cfg_degree=1,
                sp_ulysses_degree=parallelism,
                sp_ring_degree=1,
                tp_degree=1,
                shard_fn=partial(shard_model, wrap_module_names=["blocks"])
                if dit_fsdp
                else None,
                device="cuda",
                master_port=random.randint(1024, 65535),
                compile=compile,
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
        height,
        width,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        image = image.transpose(0, 1)
        y = self.vae.encode(
            [image.to(dtype=self.torch_dtype, device=self.device)],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
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
            input_video,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        frames = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return frames

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

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
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=0,
        model_type="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        attn_impl="sdpa",
        camera_movement_type: Optional[str] = None,
    ):
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
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # Initialize noise
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
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        clip_context = torch.zeros(
            (1, 257, 1280), dtype=self.torch_dtype, device=self.device
        )
        ref_latent = torch.zeros_like(
            latents, dtype=self.torch_dtype, device=self.device
        )
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            clip_context, image_latent = self.encode_image(input_image, height, width)
            ref_latent[:, :, :1] = image_latent

        if camera_movement_type is not None:
            if camera_movement_type not in CAMERA_MOVEMENT_TYPE_LIST:
                logger.error(
                    f"camera movement type {camera_movement_type} is not supported"
                )
                return None
            camera_movement_file = os.path.join(
                os.path.dirname(__file__), f"asset/{camera_movement_type}.txt"
            )
            control_camera_video = process_pose_file(
                camera_movement_file, width, height
            )
            control_camera_video = (
                control_camera_video[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
            )
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(
                        control_camera_video[:, :, 0:1], repeats=4, dim=2
                    ),
                    control_camera_video[:, :, 1:],
                ],
                dim=2,
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, 4, c, h, w)
                .transpose(2, 3)
            )
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, c * 4, h, w)
                .transpose(1, 2)
            )
            control_camera_latents = control_camera_latents.to(
                dtype=self.torch_dtype, device=self.device
            )
        else:
            control_camera_video = None
            control_camera_latents = None

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        if tea_cache_l1_thresh > 0:
            self.dit.set_tea_cache(num_inference_steps, tea_cache_l1_thresh, model_type)
        else:
            self.dit.reset_tea_cache()
        for progress_id, timestep in enumerate(
            tqdm(progress_bar_cmd(self.scheduler.timesteps), desc="generate video")
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            noise_pred_posi = self.dit.forward(
                x=input_latent,
                cond_flag=True,
                attn_impl=attn_impl,
                timestep=timestep,
                clip_feature=clip_context,
                y_camera=control_camera_latents,
                **prompt_emb_posi,
                **extra_input,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit.forward(
                    x=input_latent,
                    timestep=timestep,
                    attn_impl=attn_impl,
                    clip_feature=clip_context,
                    y_camera=control_camera_latents,
                    cond_flag=False,
                    **prompt_emb_nega,
                    **extra_input,
                )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

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
