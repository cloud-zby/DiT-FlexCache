from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.models.sd3_text_encoder import SD3TextEncoder1
from teleai_pipe.models.hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from teleai_pipe.models.hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder
from .models.model import TeleAIVideoDiTV1
from teleai_pipe.models.hunyuan_video_text_encoder import HunyuanVideoLLMEncoder
from .flow_match import FlowMatchScheduler
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from .prompter import HunyuanVideoPrompter
import torch
import random
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
import torch.nn.functional as F
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model
from teleai_pipe.pipelines.teleai_video_v1.models.model import HunyuanVideoRope
import copy


class TeleAIVideoV1Pipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler(
            shift=7.0, sigma_min=0.0, extra_one_step=True
        )
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: HunyuanVideoLLMEncoder = None
        self.dit: TeleAIVideoDiTV1 = None
        self.vae_decoder: HunyuanVideoVAEDecoder = None
        self.vae_encoder: HunyuanVideoVAEEncoder = None
        self.model_names = [
            "text_encoder_1",
            "text_encoder_2",
            "dit",
            "vae_decoder",
            "vae_encoder",
        ]
        self.dit_vram_management = False
        self.text_encoder_2_vram_management = False
        self.batch_cfg = False

    def enable_vram_management(self, with_dit=True):
        self.text_encoder_2_vram_management = True
        self.enable_cpu_offload()
        self.text_encoder_2.enable_auto_offload(
            dtype=self.torch_dtype, device=self.device
        )
        if with_dit:
            self.dit.enable_auto_offload(dtype=self.torch_dtype, device=self.device)
            self.dit_vram_management = True

    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.dit = model_manager.fetch_model("teleai_video_v1_dit")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("hunyuan_video_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        enable_vram_management=True,
        compile=False,
        dit_fsdp=False,
        parallelism: int = 1,
        use_cfg_parallel: bool = False,
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = TeleAIVideoV1Pipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        # dit modulate layer
        pipe.dit.modulate_inp_layer = copy.deepcopy(
            pipe.dit.double_blocks[0].component_a
        )
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
                shard_fn=partial(
                    shard_model, wrap_module_names=["single_blocks", "double_blocks"]
                )
                if dit_fsdp
                else None,
                device="cuda",
                master_port=random.randint(1024, 65535),
                compile=compile,
            )
        else:
            if compile:
                pipe.dit = torch.compile(pipe.dit, mode="default")
        if enable_vram_management:
            pipe.enable_vram_management(not dit_fsdp)
        return pipe

    def encode_prompt(
        self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256
    ):
        prompt_emb, pooled_prompt_emb, text_mask = self.prompter.encode_prompt(
            prompt,
            device=self.device,
            positive=positive,
            clip_sequence_length=clip_sequence_length,
            llm_sequence_length=llm_sequence_length,
        )
        return {
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "text_mask": text_mask,
        }

    def prepare_extra_input(self, latents=None, guidance=1.0):
        freqs_cos, freqs_sin = HunyuanVideoRope(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(
            device=latents.device, dtype=latents.dtype
        )
        return {"freqs_cos": freqs_cos, "freqs_sin": freqs_sin, "guidance": guidance}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, frames, tile_size=(17, 30, 30), tile_stride=(12, 20, 20)):
        tile_size = ((tile_size[0] - 1) * 4 + 1, tile_size[1] * 8, tile_size[2] * 8)
        tile_stride = (tile_stride[0] * 4, tile_stride[1] * 8, tile_stride[2] * 8)
        latents = self.vae_encoder.encode_video(
            frames, tile_size=tile_size, tile_stride=tile_stride
        )
        return latents

    def encode_image(
        self, image, height, width, tile_size=(17, 30, 30), tile_stride=(12, 20, 20)
    ):
        tile_size = ((tile_size[0] - 1) * 4 + 1, tile_size[1] * 8, tile_size[2] * 8)
        tile_stride = (tile_stride[0] * 4, tile_stride[1] * 8, tile_stride[2] * 8)
        image = self.preprocess_image(image.resize((width, height))).to(
            device=self.device
        )
        y = self.vae_encoder.encode_video(
            image.transpose(0, 1).unsqueeze(0),
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return y

    def predict_noise_with_cfg(
        self,
        input_latents,
        prompt_emb_posi,
        prompt_emb_nega,
        timestep,
        attn_impl,
        extra_input,
        cfg_scale=1.0,
    ):
        if cfg_scale == 1.0:
            return self.dit.forward(
                x=input_latents,
                t=timestep,
                attn_impl=attn_impl,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
            )
        if not self.batch_cfg:
            noise_pred_posi = self.dit.forward(
                x=input_latents,
                t=timestep,
                attn_impl=attn_impl,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
            )
            noise_pred_nega = self.dit.forward(
                x=input_latents,
                t=timestep,
                attn_impl=attn_impl,
                cond_flag=False,
                **prompt_emb_nega,
                **extra_input,
            )
            noise_pred = noise_pred_nega + cfg_scale * (
                noise_pred_posi - noise_pred_nega
            )
            return noise_pred
        else:
            prompt_emb = torch.cat(
                [prompt_emb_posi["prompt_emb"], prompt_emb_nega["prompt_emb"]], dim=0
            )
            pooled_prompt_emb = torch.cat(
                [
                    prompt_emb_posi["pooled_prompt_emb"],
                    prompt_emb_nega["pooled_prompt_emb"],
                ],
                dim=0,
            )
            text_mask = torch.cat(
                [prompt_emb_posi["text_mask"], prompt_emb_nega["text_mask"]], dim=0
            )
            input_latents = torch.cat([input_latents, input_latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            freqs_cos = torch.cat(
                [extra_input["freqs_cos"], extra_input["freqs_cos"]], dim=0
            )
            freqs_sin = torch.cat(
                [extra_input["freqs_sin"], extra_input["freqs_sin"]], dim=0
            )
            guidance = torch.cat(
                [extra_input["guidance"], extra_input["guidance"]], dim=0
            )
            noise_pred_posi, noise_pred_nega = self.dit.forward(
                x=input_latents,
                t=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                text_mask=text_mask,
                attn_impl=attn_impl,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                guidance=guidance,
            )
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
        input_video=None,
        ref_images=None,
        denoising_strength=1.0,
        seed=None,
        rand_device=None,
        height=720,
        width=1280,
        num_frames=129,
        embedded_guidance=6.0,
        cfg_scale=1.0,
        cfg_skip_ratio: float = 0,
        num_inference_steps=50,
        tea_cache_l1_thresh=0,
        tile_size=(7, 32, 32),
        tile_stride=(5, 24, 24),
        step_processor=None,
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
        attn_impl: str = "sdpa",
        model_type="t2v",
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        ## Tiler parameters
        tiler_kwargs = {"tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Initialize noise
        rand_device = self.device if rand_device is None else rand_device
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=self.torch_dtype,
        ).to(self.device)
        if input_video is not None:
            self.load_models_to_device(["vae_encoder"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        if input_image is not None:
            if model_type == "i2vhy":
                self.load_models_to_device(["vae_encoder"])
                image_latents = self.encode_image(
                    input_image, height, width, **tiler_kwargs
                ).to(dtype=self.torch_dtype, device=self.device)
                b, c, f, h, w = noise.shape
                pad_size = noise.size(2) - image_latents.size(2)
                image_latents = F.pad(
                    image_latents, (0, 0, 0, 0, 0, pad_size), mode="constant", value=0
                )
                mask_latents = torch.zeros(
                    (b, 1, f, h, w), dtype=image_latents.dtype, device=self.device
                )
                mask_latents[:, :, 0] = 1
                ref_latents = torch.cat([image_latents, mask_latents], dim=1)

        if ref_images is not None and model_type == "first_last":
            b, c, f, h, w = noise.shape
            self.load_models_to_device(["vae_encoder"])
            if not len(ref_images) == 2:
                logger.error(
                    f"FL i2v model needs 2 ref images, while we only get {len(ref_images)} image"
                )
                return None
            full_ref_images = [
                Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(num_frames)
            ]
            full_ref_images[0] = ref_images[0].resize((width, height))
            full_ref_images[-1] = ref_images[-1].resize((width, height))
            full_ref_images = self.preprocess_images(full_ref_images)
            full_ref_images = torch.stack(full_ref_images, dim=2)
            full_ref_images[:, :, 1:-1, :, :] = 0
            image_latents = self.encode_video(full_ref_images, **tiler_kwargs).to(
                dtype=self.torch_dtype, device=self.device
            )
            mask_latents = torch.ones(
                (b, 4, f, h, w), dtype=image_latents.dtype, device=self.device
            )
            mask_latents[:, :, 1:, :, :] = 0
            mask_latents[:, -1, -1, :, :] = 1
            ref_latents = torch.cat([image_latents, mask_latents], dim=1)

        # Encode prompts
        self.load_models_to_device(
            ["text_encoder_1"]
            if self.text_encoder_2_vram_management
            else ["text_encoder_1", "text_encoder_2"]
        )
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = None
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Denoise
        self.load_models_to_device([] if self.dit_vram_management else ["dit"])
        if tea_cache_l1_thresh > 0:
            self.dit.set_tea_cache(num_inference_steps, tea_cache_l1_thresh, model_type)
        for progress_id, timestep in enumerate(
            tqdm(progress_bar_cmd(self.scheduler.timesteps), desc="generate video")
        ):
            timestep = timestep.unsqueeze(0).to(self.device)
            # Inference
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                input_latents = latents
                if input_image is not None or ref_images is not None:
                    input_latents = torch.cat([latents, ref_latents], dim=1)
                if (
                    progress_id > cfg_skip_ratio * num_inference_steps
                    and not self.batch_cfg
                ):
                    cfg_scale = 1.0
                noise_pred = self.predict_noise_with_cfg(
                    input_latents=input_latents,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    cfg_scale=cfg_scale,
                    attn_impl=attn_impl,
                    extra_input=extra_input,
                )

            # (Experimental feature, may be removed in the future)
            if step_processor is not None:
                self.load_models_to_device(["vae_decoder"])
                rendered_frames = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    latents,
                    to_final=True,
                )
                rendered_frames = self.vae_decoder.decode_video(
                    rendered_frames, **tiler_kwargs
                )
                rendered_frames = self.tensor2video(rendered_frames[0])
                rendered_frames = step_processor(
                    rendered_frames, original_frames=input_video
                )
                self.load_models_to_device(["vae_encoder"])
                rendered_frames = self.preprocess_images(rendered_frames)
                rendered_frames = torch.stack(rendered_frames, dim=2)
                target_latents = self.encode_video(rendered_frames).to(
                    dtype=self.torch_dtype, device=self.device
                )
                noise_pred = self.scheduler.return_to_timestep(
                    self.scheduler.timesteps[progress_id], latents, target_latents
                )
                self.load_models_to_device([] if self.dit_vram_management else ["dit"])

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents
            )
        # Decode
        self.load_models_to_device(["vae_decoder"])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames

    def __del__(self):
        del self.dit
