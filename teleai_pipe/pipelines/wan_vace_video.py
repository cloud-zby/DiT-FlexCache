from teleai_pipe.common.model_manager import ModelManager
import torch
from einops import rearrange
from tqdm import tqdm
from loguru import logger
from ..models.wan_video_vace import VaceWanModel
from .wan_video import WanVideoPipeline


class WanVaceVideoPipeline(WanVideoPipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        tokenizer_path=None,
        vae_dtype=torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            tokenizer_path=tokenizer_path,
            vae_dtype=vae_dtype,
        )
        self.vace: VaceWanModel = True

    def fetch_models(self, model_manager: ModelManager):
        super().fetch_models(model_manager)
        self.dit.vace = model_manager.fetch_model("wan_video_vace")

    def prepare_vace_kwargs(
        self,
        vace_video=None,
        vace_mask=None,
        vace_reference_image=None,
        height=480,
        width=832,
        num_frames=81,
        seed=None,
        rand_device="cpu",
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        if (
            vace_video is not None
            or vace_mask is not None
            or vace_reference_image is not None
        ):
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )

            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )

            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            logger.info("encode inactive")
            inactive = self.encode_video(
                inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=self.torch_dtype, device=self.device)
            logger.info("encode reactive")
            reactive = self.encode_video(
                reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            ).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)  # channel 32

            vace_mask_latents = rearrange(
                vace_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is not None:
                vace_reference_image = self.preprocess_images(
                    [vace_reference_image.resize((width, height))]
                )
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(
                    dtype=self.torch_dtype, device=self.device
                )
                vace_reference_latents = self.encode_video(
                    vace_reference_image,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                    dim=1,
                )
                vace_video_latents = torch.concat(
                    (vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents),
                    dim=2,
                )

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return vace_context
        else:
            return None

    def predict_noise_with_cfg(
        self,
        latents,
        timestep,
        prompt_emb_posi,
        prompt_emb_nega,
        extra_input,
        attn_impl,
        vace_context,
        cfg_scale=1.0,
        vace_scale=1.0,
    ):
        if cfg_scale == 1.0:
            return self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                vace_scale=vace_scale,
                vace_context=vace_context,
            )
        if not self.batch_cfg:
            noise_pred_posi = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                **prompt_emb_posi,
                **extra_input,
                attn_impl=attn_impl,
                vace_scale=vace_scale,
                vace_context=vace_context,
            )
            noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                **prompt_emb_nega,
                **extra_input,
                attn_impl=attn_impl,
                vace_scale=vace_scale,
                vace_context=vace_context,
            )
            noise_pred = noise_pred_nega + cfg_scale * (
                noise_pred_posi - noise_pred_nega
            )
            return noise_pred
        else:
            context = torch.cat(
                [prompt_emb_posi["context"], prompt_emb_nega["context"]], dim=0
            )
            if clip_feature is not None:
                clip_feature = torch.cat([clip_feature, clip_feature], dim=0)

            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = self.dit.forward(
                x=latents,
                timestep=timestep,
                context=context,
                clip_feature=clip_feature,
                attn_impl=attn_impl,
                **extra_input,
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
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
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
        attn_impl="sdpa",
        enable_riflex=False,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
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

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = None
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input

        # vace
        vace_context = self.prepare_vace_kwargs(
            vace_video,
            vace_video_mask,
            vace_reference_image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device=rand_device,
            **tiler_kwargs,
        )
        # Initialize noise
        noise = self.generate_noise(
            (1, 16, vace_context.shape[2], height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        if enable_riflex:
            self.dit.enable_riflex(k=6, L_test=noise.shape[2])
        else:
            self.dit.disable_riflex()
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise
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

            if (
                progress_id > (1 - cfg_skip_ratio) * num_inference_steps
                and not self.batch_cfg
            ):
                cfg_scale = 1.0
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    extra_input=extra_input,
                    attn_impl=attn_impl,
                    cfg_scale=cfg_scale,
                    vace_context=vace_context,
                    vace_scale=vace_scale,
                )

            # Scheduler
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents
            )
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]
        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
