from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..prompters.wan_prompter import WanPrompter
import torch
import os
import math
from einops import rearrange
import numpy as np
from PIL import Image, ImageOps
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
from ..schedulers.flow_match_v2 import FlowMatchSchedulerV2
from ..models.wan_video_dit import RMSNorm
from ..models.wan_lora import WanLoraWrapper
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from .teleai_video_sr import TeleVideoSRPipeline

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
    aspect_ratio, resolution="480p", height_division_factor=16, width_division_factor=16
):
    if aspect_ratio not in SUPPORTED_ASPECT_RATIO_LIST:
        logger.error(f"aspect ratio {aspect_ratio} is not supported")
    w, h = aspect_ratio.split(":")
    return get_target_image_size(
        int(w), int(h), resolution, height_division_factor, width_division_factor
    )

def pad_ref_images(ref_imgs, w, h):

    ref_images = []
    for img in ref_imgs:
        # Calculate the required size to keep aspect ratio and fill the rest with padding.
        img_ratio = img.width / img.height
        target_ratio = w / h
        
        if img_ratio > target_ratio:  # Image is wider than target
            new_width = w
            new_height = int(new_width / img_ratio)
        else:  # Image is taller than target
            new_height = h
            new_width = int(new_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with the target size and place the resized image in the center
        delta_w = w - img.size[0]
        delta_h = h - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        ref_images.append(new_img)
        
    return ref_images

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


class TeleAIVideoRefV1(BasePipeline):
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
        self.model_names = ["text_encoder", "dit", "vae", "image_encoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dit_parallel = False
        self.batch_cfg = False
        self.vae_dtype = vae_dtype
        self.vae_parallel = False
        self.sr_pipe: TeleVideoSRPipeline = None

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
        pipe.dit_parallel = parallelism > 2 
        if parallelism > 2: 
            if use_cfg_parallel:
                pipe.batch_cfg = True
                pipe.dit.enable_cfgp()
                cfg_degree = 3 
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

    def get_vae_latents(
        self, 
        ref_images, 
        device,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        ref_vae_latents = []
        for ref_image in ref_images:
            ref_image = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device)
            img_vae_latent = self.vae.encode(
                [ref_image.unsqueeze(1).to(dtype=self.vae_dtype)], 
                device=device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            ref_vae_latents.append(img_vae_latent[0])
                    
        return torch.cat(ref_vae_latents, dim=1)

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
        dit,
        latents,
        ref_latents,
        ref_latent_neg,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        # clip_feature,
        extra_input,
        attn_impl,
        guide_scale_text=1.0,
        guide_scale_img=1.0,
        use_cfg_zero_star=False,
    ):
        if guide_scale_text == 1.0 and guide_scale_img == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                # clip_feature=clip_feature,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )

        input_latents_pos = [torch.cat([latent[:,:-ref_latent.shape[1]], ref_latent], dim=1) for latent, ref_latent in zip(latents, ref_latents)]
        input_latents_neg = [torch.cat([latent[:,:-ref_latent_neg.shape[1]], ref_latent_neg], dim=1) for latent, ref_latent_neg in zip(latents, ref_latent_neg)]
            
        input_latents_pos = torch.stack(input_latents_pos, dim=0)
        input_latents_neg = torch.stack(input_latents_neg, dim=0)

        if not self.batch_cfg:
            pos_it = dit.forward(
                x=input_latents_pos,
                timestep=timestep,
                cond_flag=True,
                # clip_feature=clip_feature,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
            )
            pos_i = dit.forward(
                x=input_latents_pos,
                timestep=timestep,
                cond_flag=True,
                # clip_feature=clip_feature,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
            )
            neg = dit.forward(
                x=input_latents_neg,
                timestep=timestep,
                cond_flag=False,
                # clip_feature=clip_feature,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
            )
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"], prompt_emb_nega["context"]], dim=0
            )
            # context = torch.cat(
            #     [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            # )
            # if clip_feature is not None:
            #     clip_feature = torch.cat([clip_feature, clip_feature], dim=0)

            # latents = torch.cat([input_latents_pos, input_latents_neg], dim=0)
            latents = torch.cat([input_latents_pos, input_latents_pos, input_latents_neg], dim=0)
            timestep = torch.cat([timestep, timestep, timestep], dim=0)

            pos_it, pos_i, neg = dit.forward(
                x=latents.to(dtype=self.torch_dtype, device=self.device),
                timestep=timestep.to(dtype=self.torch_dtype, device=self.device),
                context=context.to(dtype=self.torch_dtype, device=self.device),
                # clip_feature=clip_feature,
                attn_impl=attn_impl,
                **extra_input,
            )
            # pos_i = pos_it

        if use_cfg_zero_star:
            assert use_cfg_zero_star == False, "not check yet"
            # TODO check this
            pos_it_flat = pos_it.view(1, -1)
            pos_i_flat = pos_i.view(1, -1)
            neg_flat = neg.view(1, -1)
            alpha1 = optimized_scale(pos_it_flat, pos_i_flat)
            alpha2 = optimized_scale(pos_i_flat, neg_flat)
            alpha1 = alpha1.view(1, *([1] * (len(pos_it.shape) - 1)))
            alpha2 = alpha2.view(1, *([1] * (len(pos_i.shape) - 1)))
            alpha1 = alpha1.to(pos_it.dtype)
            alpha2 = alpha2.to(pos_i.dtype)
            noise_pred = neg * alpha1 + guide_scale_img * (pos_i - neg * alpha1) + guide_scale_text * (pos_it - pos_i * alpha2)

        else:
            noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i)
                
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        ref_images=[],
        end_image=None,
        seed=None,
        rand_device: Optional[str] = None,
        height=480,
        width=832,
        num_frames=81,
        base_cfg_scale=5.5,
        base_cfg_scale_img=5.5,
        sr_cfg_scale=5.5,
        base_num_inference_steps=40,
        sr_num_inference_steps=10,
        base_sigma_shift=5.0,
        sr_sigma_shift=5.0,
        base_timestep_range=(333, 1000),
        sr_timestep_range=(0, 500),
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        model_type="",
        progress_bar_cmd=tqdm,
        attn_impl="sdpa",
        zero_init_steps=-1,
        sr_scale: int = 2,
        enable_dit_magcache: bool = False,
        enable_sr_magcache: bool = False,
    ):
        # Parameter check
        height = int(height // sr_scale)
        width = int(width // sr_scale)
        height, width = self.check_resize_height_width(height, width)
        # resize ref images
        ref_images = pad_ref_images(ref_images, width, height)

        logger.info(
            f"start genereate {num_frames} {width * sr_scale}x{height * sr_scale} frames"
        )
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

        self.load_models_to_device(["vae"])
        ref_latents = [self.get_vae_latents(ref_images, self.device)]
        ref_latents_neg = [torch.zeros_like(ref_latents[0])]

        F = num_frames
        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)
        sp_size = 1

        target_shape = (
            1, 
            self.vae.model.z_dim, 
            (F - 1) // vae_stride[0] + 1 + ref_latents[0].shape[1], 
            height // vae_stride[1], 
            width // vae_stride[2]
        )
        # seq_len = math.ceil((target_shape[3] * target_shape[4]) / (patch_size[1] * patch_size[2]) * target_shape[2] / sp_size) * sp_size

        noise = self.generate_noise(
            target_shape,
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
        clip_context = None
        ref_latent = None
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            clip_context, ref_latent = self.encode_image(
                input_image, end_image, num_frames, height, width
            )

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # base Denoise
        self.load_models_to_device(["dit"])

        # Scheduler
        if sr_scale == 1:
            self.scheduler.sigma_min = 0
        else:
            self.scheduler.sigma_min = base_timestep_range[0] / 1000
        self.scheduler.sigma_max = 1
        self.scheduler.set_timesteps(
            base_num_inference_steps,
            shift=base_sigma_shift,
        )
        if enable_dit_magcache:
            self.dit.set_mag_cache(
                base_num_inference_steps,
                model_type=model_type,
            )
        else:
            self.dit.reset_mag_cache()

        for progress_id, timestep in enumerate(
            tqdm(
                progress_bar_cmd(self.scheduler.timesteps), desc="denoise with base dit"
            )
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            input_latent = latents
            # if ref_latent is not None:
            #     input_latent = torch.cat([input_latent, ref_latent], dim=1)
            if progress_id <= zero_init_steps:
                noise_pred = latents * 0
            else:
                with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                    noise_pred = self.predict_noise_with_cfg(
                        dit=self.dit,
                        latents=input_latent,
                        ref_latents=ref_latents,
                        ref_latent_neg=ref_latents_neg,
                        timestep=timestep,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        # clip_feature=clip_context,
                        extra_input=extra_input,
                        attn_impl=attn_impl,
                        guide_scale_text=base_cfg_scale,
                        guide_scale_img=base_cfg_scale_img,
                        use_cfg_zero_star=zero_init_steps >= 0,
                    ).to(input_latent.device)

            # Scheduler
            latents = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                latents,
                denoising_strength=1.0,
            )
        x0 = latents
        x0 = [x0_[:,:-ref_latents[0].shape[1]] for x0_ in x0]
        latents = torch.stack(x0, dim=0)

        if sr_scale > 1 and self.sr_pipe is not None:
            latents = self.sr_pipe.sr_process_tensor(
                latents,
                self.scheduler.sigma_min,
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
            )
        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        return frames

    def __del__(self):
        del self.dit
        del self.sr_pipe
        del self.vae
