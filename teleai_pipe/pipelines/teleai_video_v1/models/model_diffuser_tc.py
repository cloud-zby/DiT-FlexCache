from .model_diffusers import HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from loguru import logger
from teleai_pipe.plugin.teacache import TeaCache


class HunyuanVideoTeaCache(HunyuanVideoTransformer3DModel):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            num_attention_heads,
            attention_head_dim,
            num_layers,
            num_single_layers,
            num_refiner_layers,
            mlp_ratio,
            patch_size,
            patch_size_t,
            qk_norm,
            guidance_embeds,
            text_embed_dim,
            pooled_projection_dim,
            rope_theta,
            rope_axes_dim,
        )
        self.cond_tea_cache = None
        self.uncond_tea_cache = None

    def set_tea_cache(self, num_steps, th, model_type):
        from teleai_pipe.plugin.teacache.teacache_coeff import TELEAI_VIDEO_V1_COEFF_MAP

        logger.info(
            f"init teacache for task {model_type} {str(TELEAI_VIDEO_V1_COEFF_MAP[model_type])} with th {th}"
        )
        self.cond_tea_cache = TeaCache(
            num_inference_steps=num_steps,
            rel_l1_thresh=th,
            coefficients=TELEAI_VIDEO_V1_COEFF_MAP[model_type],
        )
        self.uncond_tea_cache = TeaCache(
            num_inference_steps=num_steps,
            rel_l1_thresh=th,
            coefficients=TELEAI_VIDEO_V1_COEFF_MAP[model_type],
        )

    def reset_tea_cache(self):
        self.cond_tea_cache = None
        self.uncond_tea_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cond_flag: bool = True,
        attn_impl: str = "sdpa",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        effective_condition_sequence_length = encoder_attention_mask.sum(
            dim=1, dtype=torch.int
        )  # [B,]
        effective_sequence_length = (
            latent_sequence_length + effective_condition_sequence_length
        )
        encoder_hidden_states = encoder_hidden_states[
            :, :effective_condition_sequence_length, :
        ]

        maual_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.transformer_blocks[0].norm1(hidden_states, emb=temb)
        )
        attention_mask = None

        if self.cond_tea_cache is None and self.uncond_tea_cache is None:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    attn_impl=attn_impl,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    attn_impl=attn_impl,
                )
        else:
            if cond_flag:
                tea_cache = self.cond_tea_cache
            else:
                tea_cache = self.uncond_tea_cache
            tea_cache_update = tea_cache.check(hidden_states, maual_hidden_states)
            if tea_cache_update:
                hidden_states = tea_cache.update(hidden_states)
            else:
                # 4. Transformer blocks
                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        attn_impl=attn_impl,
                    )

                for block in self.single_transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        attn_impl=attn_impl,
                    )
                tea_cache.store(hidden_states)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
