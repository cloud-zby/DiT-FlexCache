from teleai_pipe.common.model_manager import ModelManager
import torch
import os
from einops import rearrange
import numpy as np
from tqdm import tqdm
from typing import Optional
import torch.nn as nn
from ..pipelines.wan_video import WanVideoPipeline
from typing import Callable, Dict, List, Union, Optional

# parallel
import random
from functools import partial
from teleai_pipe.distributed.parallel import ParallelModel, shard_model


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


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


class WanVideo3DS2VPipeline(WanVideoPipeline):
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

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
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

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
            nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
        )
        state_dict_new = {}
        for key in model_manager.state_dict_new_module:
            if "dwpose_embedding" in key:
                state_dict_new[key.split("dwpose_embedding.")[1]] = (
                    model_manager.state_dict_new_module[key]
                )
        self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

        state_dict_new = {}
        for key in model_manager.state_dict_new_module:
            if "randomref_embedding_pose" in key:
                state_dict_new[key.split("randomref_embedding_pose.")[1]] = (
                    model_manager.state_dict_new_module[key]
                )
        self.randomref_embedding_pose.load_state_dict(state_dict_new, strict=True)

        # update self attentions parameters.
        if model_manager.state_dict_self_attn:
            # print(model_manager.state_dict_self_attn.keys())
            original_state_dict = self.dit.state_dict()
            updated_state_dict = {
                key: model_manager.state_dict_self_attn[("pipe.dit." + key)]
                for key in original_state_dict.keys()
                if ("pipe.dit." + key) in model_manager.state_dict_self_attn
            }

            original_state_dict.update(updated_state_dict)
            self.dit.load_state_dict(original_state_dict)

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        compile=False,
        dit_fsdp=False,
        parallelism: int = 1,
        vae_parallel: bool = False,
        dit_device_ids : Optional[List] = None,
        vae_device_ids :Optional[List] = None,
        enable_vram_management = False,
        use_cfg_parallel: bool = False,
        use_usp: bool = False,
    ):
    # def from_model_manager(
    #     model_manager: ModelManager,
    #     torch_dtype=None,
    #     device=None,
    #     compile=False,
    #     dit_fsdp=False,
    #     parallelism: int = 1,
    # ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideo3DS2VPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        pipe.vae_parallel = vae_parallel
        pipe.dit_parallel = parallelism > 1
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


        if vae_parallel:
            vae_parallel_num = (
                len(vae_device_ids) if vae_device_ids else parallelism
            )
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

        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
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
        tea_cache_l1_thresh=None,
        model_type="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        dwpose_data=None,
        random_ref_dwpose=None,
        context_size=21,
        context_overlap=4,
        attn_impl="sdpa",
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        #
        if num_frames % 4 != 1:
            num_frames = (num_frames - 1) // 4 * 4 + 1
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
        # context_size = 21
        # context_overlap = 4
        context_queue = list(
            context_scheduler(
                0,
                31,
                noise.shape[2],
                context_size=context_size,
                context_stride=1,
                context_overlap=context_overlap,
            )
        )
        context_step = min(1, int(np.ceil(np.log2(noise.shape[2] / context_size))) + 1)
        num_frames = noise.shape[2]
        context_queue[-1] = [
            e % num_frames
            for e in range(
                num_frames - context_size * context_step, num_frames, context_step
            )
        ]
        import math

        context_batch_size = 1
        num_context_batches = math.ceil(len(context_queue) / context_batch_size)
        global_context = []
        for i in range(num_context_batches):
            global_context.append(
                context_queue[i * context_batch_size : (i + 1) * context_batch_size]
            )

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        clip_context = None
        ref_latent = None
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            clip_context, ref_latent = self.encode_image(
                input_image, None, context_size * 4 - 3, height, width
            )
        else:
            image_emb = {}

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        if tea_cache_l1_thresh > 0:
            self.dit.set_tea_cache(num_inference_steps, tea_cache_l1_thresh, model_type)
        else:
            self.dit.reset_tea_cache()
        self.dwpose_embedding.to(self.device)
        self.randomref_embedding_pose.to(self.device)
        dwpose_data = dwpose_data.unsqueeze(0)

        # noise_per = self.generate_noise((1, 16, 21, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        dwpose_data_list = []
        #
        first_feature_per_seg = []
        for ii in global_context:
            dwpose_data_per = dwpose_data[
                :, :, (ii[0][0] * 4) : (ii[0][-1] * 4 + 1), :, :
            ]
            dwpose_data_list.append(
                self.dwpose_embedding(
                    (
                        torch.cat(
                            [
                                dwpose_data_per[:, :, :1].repeat(1, 1, 3, 1, 1),
                                dwpose_data_per,
                            ],
                            dim=2,
                        )
                        / 255.0
                    ).to(self.device)
                ).to(torch.bfloat16)
            )
            #
            first_feature_per_seg.append(
                torch.randn_like(latents[:, :, ii[0][0] : (ii[0][0] + 2)])
            )

        random_ref_dwpose_data = (
            self.randomref_embedding_pose(
                (random_ref_dwpose.unsqueeze(0) / 255.0)
                .to(self.device)
                .permute(0, 3, 1, 2)
            )
            .unsqueeze(2)
            .to(torch.bfloat16)
        )  # [1, 20, 104, 60]

        ref_latent = ref_latent + random_ref_dwpose_data
        # condition = rearrange(dwpose_data, 'b c f h w -> b (f h w) c').contiguous()

        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )
            noise_pred_out = torch.zeros_like(latents)
            for i_index, context in enumerate(global_context):
                latent_model_input = torch.cat([latents[:, :, c] for c in context])
                # latent_model_input[:,:,:1] = first_feature_per_seg[i_index]
                latent_model_input[:, :, :2] = first_feature_per_seg[i_index]
                condition = rearrange(
                    dwpose_data_list[i_index], "b c f h w -> b (f h w) c"
                ).contiguous()
                input_latent = latent_model_input
                if ref_latent is not None:
                    input_latent = torch.cat([input_latent, ref_latent], dim=1)

                noise_pred_posi = self.dit.forward(
                    x=input_latent,
                    timestep=timestep,
                    attn_impl=attn_impl,
                    cond_flag=True,
                    add_condition=condition,
                    clip_feature=clip_context,
                    **prompt_emb_posi,
                    **extra_input,
                )
                # Inference
                #
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit.forward(
                        x=input_latent,
                        timestep=timestep,
                        attn_impl=attn_impl,
                        cond_flag=False,
                        add_condition=condition,
                        clip_feature=clip_context,
                        **prompt_emb_nega,
                        **extra_input,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                noise_pred = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    latent_model_input,
                )
                #
                for j, c in enumerate(context):
                    # if not (i_index !=0 and j==0):
                    if i_index == 0 and j == 0:
                        counter[:, :, c] = counter[:, :, c] + 1
                        noise_pred_out[:, :, c] = (
                            noise_pred_out[:, :, c] + noise_pred[j : j + 1]
                        )
                    else:
                        # skip the first feature
                        # c = c[1:]
                        # counter[:, :, c] = counter[:, :, c] + 1
                        # noise_pred_out[:, :, c] = noise_pred_out[:, :, c] + noise_pred[j:j+1,:,1:]
                        c = c[2:]
                        counter[:, :, c] = counter[:, :, c] + 1
                        noise_pred_out[:, :, c] = (
                            noise_pred_out[:, :, c] + noise_pred[j : j + 1, :, 2:]
                        )

                first_feature_per_seg[i_index] = noise_pred[:, :, :2]

            latents = noise_pred_out / counter
            # Scheduler
            # latents = self.scheduler.step(noise_pred_out, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
