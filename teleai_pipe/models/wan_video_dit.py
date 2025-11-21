import torch
import torch.nn as nn
import math
import torch.amp as amp
from loguru import logger
from typing import Tuple, Optional
from einops import rearrange
from teleai_pipe.utils.utils import hash_state_dict_keys
from teleai_pipe.models.attention import attention as attn_func
from teleai_pipe.plugin.teacache import TeaCache
from teleai_pipe.plugin.magcache import MagCache
from teleai_pipe.models.attention import long_context_attention as long_attn_func
from teleai_pipe.models.sparse_attention import sparse_attention as sparse_attn_func
from teleai_pipe.models.sparse_attention import (
    sparse_long_context_attention as sparse_long_attn_func,
)
from teleai_pipe.models.sparse_attention import SparseAttentionParams
from teleai_pipe.distributed.parallel import get_cfg_rank
from teleai_pipe.distributed.parallel import (
    sequence_parallel,
    sequence_parallel_unshard,
)
from teleai_pipe.plugin.riflex import get_1d_rotary_pos_embed_riflex
from teleai_pipe.vram_management.manager import (
    WeightAsyncStreamManager,
)
from teleai_pipe.vram_management.weight_utils import to_pin_memory
from teleai_pipe.vram_management.weight_utils import (
    add_weight_management_methods,
    add_phase_management_to_block,
    create_phase_config,
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return [f_freqs_cis, h_freqs_cis, w_freqs_cis]


@amp.autocast("cuda", enabled=False)
@torch.compiler.disable()
def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


@amp.autocast("cuda", enabled=False)
@torch.compiler.disable()
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x,
        freqs,
        attn_impl,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        if getattr(self, "use_usp", False):
            if sparse_params is not None and sparse_params.sparse_impl == "radial":
                x = sparse_long_attn_func(
                    q,
                    k,
                    v,
                    input_layout="BNSD",
                    output_layout="BNSD",
                    sparse_params=sparse_params,
                )
            else:
                x = long_attn_func(
                    q,
                    k,
                    v,
                    attn_impl=attn_impl,
                    input_layout="BNSD",
                    output_layout="BNSD",
                )
        elif sparse_params is not None and sparse_params.sparse_impl == "radial":
            batch_size = q.shape[0]
            seqlen = q.shape[2]
            q = rearrange(q, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)
            k = rearrange(k, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)
            v = rearrange(v, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)

            x = sparse_attn_func(
                q,
                k,
                v,
                input_layout="BSND",
                output_layout="BSND",
                sparse_params=sparse_params,
            )

            x = rearrange(
                x.unsqueeze(0), "b s n d -> b n s d", s=seqlen, n=self.num_heads
            )
        else:
            x = attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BNSD", output_layout="BNSD"
            )
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)

        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_impl: str):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        if getattr(self, "use_usp", False):
            x = long_attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BNSD", output_layout="BNSD"
            )
        else:
            x = attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BNSD", output_layout="BNSD"
            )
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            k_img = rearrange(k_img, "b s (n d) -> b n s d", n=self.num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b n s d", n=self.num_heads)
            if getattr(self, "use_usp", False):
                y = long_attn_func(
                    q,
                    k_img,
                    v_img,
                    attn_impl=attn_impl,
                    input_layout="BNSD",
                    output_layout="BNSD",
                )
            else:
                y = attn_func(
                    q,
                    k_img,
                    v_img,
                    attn_impl=attn_impl,
                    input_layout="BNSD",
                    output_layout="BNSD",
                )
            y = rearrange(y, "b n s d -> b s (n d)", n=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def enable_usp(self):
        setattr(self, "use_usp", True)
        setattr(self.self_attn, "use_usp", True)
        setattr(self.cross_attn, "use_usp", True)

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_impl,
        sparse_params: SparseAttentionParams = None,
    ):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if sparse_params is not None:
            attn_output = self.self_attn(
                input_x, freqs, attn_impl=attn_impl, sparse_params=sparse_params
            )
        else:
            attn_output = self.self_attn(input_x, freqs, attn_impl=attn_impl)
        x = self.gate(x, gate_msa, attn_output)
        x = x + self.cross_attn(self.norm3(x), context, attn_impl=attn_impl)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        add_control_adapter=False,
        in_dim_control_adapter=24,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.head_dim = dim // num_heads
        self.normal_freqs = precompute_freqs_cis_3d(self.head_dim)
        self.freqs = self.normal_freqs
        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if add_control_adapter:
            from teleai_pipe.pipelines.teleai_video_v1_4.models.wan_camera_adapter import (
                SimpleAdapter,
            )

            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None
        self.has_image_pos_emb = has_image_pos_emb

        self.dtype = torch.bfloat16
        self.cond_tea_cache = None
        self.uncond_tea_cache = None
        self.cond_mag_cache = None
        self.uncond_mag_cache = None
        self.vace = None
        self.fsdp_flag = False
        # async swap
        self.weights_stream_mgr = None
        self.infer_func = self._infer_without_offload
        self.async_swap_flag = False
        self.clean_cuda_cache = True

    def enable_async_swap(self, config):
        self.async_swap_flag = True
        self.async_swap_config = config
        to_pin_memory(self.blocks)
        self._prepare_weight_management()
        self._init_offload_manager()

    def _prepare_weight_management(self):
        wan_phase_config = create_phase_config(
            {
                "modulation": ["modulation"],
                "self_attn": ["norm1", "self_attn"],
                "cross_attn": ["norm3", "cross_attn"],
                "ffn": ["norm2", "ffn", "gate"],
            }
        )

        for block in self.blocks:
            add_phase_management_to_block(block, phase_config=wan_phase_config)
        add_weight_management_methods(self)

    def _init_offload_manager(self):
        blocks_num = len(self.blocks)
        phases_num = 4  # modulation, self_attn, cross_attn, ffn
        config = self.async_swap_config
        offload_granularity = config.get("offload_granularity", "block")
        offload_ratio = config.get("offload_ratio", 1.0)

        if offload_granularity == "block":
            self.infer_func = self._infer_with_block_offload
        elif offload_granularity == "phase":
            self.infer_func = self._infer_with_phases_offload

        self.weights_stream_mgr = WeightAsyncStreamManager(
            blocks_num=blocks_num,
            offload_ratio=offload_ratio,
            phases_num=phases_num,
        )

    def forward_blocks(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_impl,
        vace_context=None,
        vace_hints=None,
        vace_scale=None,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        return self.infer_func(
            x,
            context,
            t_mod,
            freqs,
            attn_impl,
            vace_context,
            vace_hints,
            vace_scale,
            sparse_params,
        )

    @torch._dynamo.disable
    def _infer_with_block_offload(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_impl,
        vace_context=None,
        vace_hints=None,
        vace_scale=None,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        for block_idx, block in enumerate(self.blocks):
            if sparse_params is not None:
                sparse_params.update(layer_idx=block_idx)
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = block
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < len(self.blocks) - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, self.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.weights_stream_mgr.active_weights[0](
                    x, context, t_mod, freqs, attn_impl, sparse_params=sparse_params
                )
                if (
                    vace_context is not None
                    and block_idx in self.vace.vace_layers_mapping
                ):
                    x = (
                        x
                        + vace_hints[self.vace.vace_layers_mapping[block_idx]]
                        * vace_scale
                    )

            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_with_phases_offload(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_impl,
        vace_context=None,
        vace_hints=None,
        vace_scale=None,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        for block_idx, block in enumerate(self.blocks):
            if sparse_params is not None:
                sparse_params.update(layer_idx=block_idx)
            for phase_idx in range(self.weights_stream_mgr.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    phase = block.compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    cur_phase_idx, cur_phase = self.weights_stream_mgr.active_weights[0]

                    if cur_phase_idx == 0:
                        (
                            shift_msa,
                            scale_msa,
                            gate_msa,
                            shift_mlp,
                            scale_mlp,
                            gate_mlp,
                        ) = (
                            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                            + t_mod
                        ).chunk(6, dim=1)

                    elif cur_phase_idx == 1:
                        input_x = modulate(block.norm1(x), shift_msa, scale_msa)
                        y_out = block.self_attn(
                            input_x,
                            freqs,
                            attn_impl=attn_impl,
                            sparse_params=sparse_params,
                        )

                    elif cur_phase_idx == 2:
                        x = block.gate(x, gate_msa, y_out)
                        attn_out = block.cross_attn(
                            block.norm3(x), context, attn_impl=attn_impl
                        )
                        x = x + attn_out

                    elif cur_phase_idx == 3:
                        input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
                        ffn_out = block.ffn(input_x)
                        x = block.gate(x, gate_mlp, ffn_out)

                if phase_idx == self.weights_stream_mgr.phases_num - 1:
                    if (
                        vace_context is not None
                        and block_idx in self.vace.vace_layers_mapping
                    ):
                        x = (
                            x
                            + vace_hints[self.vace.vace_layers_mapping[block_idx]]
                            * vace_scale
                        )

                is_last_phase = (
                    block_idx == self.weights_stream_mgr.block_nums - 1
                    and phase_idx == self.weights_stream_mgr.phases_num - 1
                )
                if not is_last_phase:
                    next_block_idx = (
                        block_idx + 1
                        if phase_idx == self.weights_stream_mgr.phases_num - 1
                        else block_idx
                    )
                    next_phase_idx = (
                        phase_idx + 1
                    ) % self.weights_stream_mgr.phases_num
                    self.weights_stream_mgr.prefetch_phase(
                        next_block_idx, next_phase_idx, self.blocks
                    )

                self.weights_stream_mgr.swap_phases()

            if self.clean_cuda_cache:
                del input_x, y_out, attn_out, ffn_out
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            del shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            del context, t_mod, freqs, attn_impl
            torch.cuda.empty_cache()

        return x

    def enable_cfgp(self):
        logger.info("wan dit enable cfgp")
        setattr(self, "use_cfgp", True)

    def enable_usp(self):
        logger.info("wan dit enable usp")
        setattr(self, "use_usp", True)
        for block in self.blocks:
            block.enable_usp()

    def set_mag_cache(self, num_steps, model_type, init_step: int = 0):
        from teleai_pipe.plugin.magcache.magcache_params import WAN_VIDEO_PARAM_MAP

        mag_cache_params = WAN_VIDEO_PARAM_MAP[model_type]
        logger.info(
            f"init magcache for task {model_type} {str} with param {mag_cache_params}"
        )
        self.cond_mag_cache = MagCache(
            num_inference_steps=num_steps,
            K=mag_cache_params["K"],
            thresh=mag_cache_params["thresh"],
            retention_ratio=mag_cache_params["retention_ratio"],
            mag_ratios=mag_cache_params["cond_mag_ratios"],
            init_step=init_step,
        )
        self.uncond_mag_cache = MagCache(
            num_inference_steps=num_steps,
            K=mag_cache_params["K"],
            thresh=mag_cache_params["thresh"],
            retention_ratio=mag_cache_params["retention_ratio"],
            mag_ratios=mag_cache_params["uncond_mag_ratios"],
            init_step=init_step,
        )

    def set_tea_cache(self, num_steps, th, model_type):
        from teleai_pipe.plugin.teacache.teacache_coeff import WAN_VIDEO_COEFF_MAP

        coeff = WAN_VIDEO_COEFF_MAP[model_type]
        logger.info(f"init teacache for task {model_type} {str} with th {th}")
        self.cond_tea_cache = TeaCache(
            num_inference_steps=num_steps,
            rel_l1_thresh=th,
            coefficients=coeff,
            start_step=5,
        )
        self.uncond_tea_cache = TeaCache(
            num_inference_steps=num_steps,
            rel_l1_thresh=th,
            coefficients=coeff,
            start_step=5,
        )

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def onload_device(self, device):
        if hasattr(self, "vram_management_enabled") and self.vram_management_enabled:
            for module in self.modules():
                if hasattr(module, "onload"):
                    module.onload()
        else:
            if self.async_swap_flag:
                for name, module in self.named_children():
                    if name == "blocks":
                        continue
                    else:
                        module.to(device)
            else:
                self.to(device)

    def offload_device(self):
        if hasattr(self, "vram_management_enabled") and self.vram_management_enabled:
            for module in self.modules():
                if hasattr(module, "offload"):
                    module.offload()
        else:
            if self.async_swap_flag:
                for name, module in self.named_children():
                    if name == "blocks":
                        continue
                    else:
                        module.cpu()
            else:
                self.cpu()

    def enable_riflex(
        self,
        k=6,
        L_test=66,
        L_test_scale=4.886,
    ):
        self.freqs[0] = get_1d_rotary_pos_embed_riflex(
            1024,
            self.head_dim - 4 * (self.head_dim // 6),
            use_real=False,
            k=k,
            L_test=L_test,
            L_test_scale=L_test_scale,
        )

    def disable_riflex(self):
        self.freqs = self.normal_freqs

    def compile(self, *args, **kwarg):
        """重写compile方法实现选择性编译"""
        original_fn = self.__class__.forward_blocks.__get__(self)
        self.forward_blocks = torch.compile(original_fn, *args, **kwarg)

    def _infer_without_offload(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_impl,
        vace_context=None,
        vace_hints=None,
        vace_scale=None,
        sparse_params: Optional[SparseAttentionParams] = None,
    ):
        for block_id, block in enumerate(self.blocks):
            if sparse_params is not None:
                sparse_params.update(layer_idx=block_id)
            x = block(
                x,
                context,
                t_mod,
                freqs,
                attn_impl=attn_impl,
                sparse_params=sparse_params,
            )
            if vace_context is not None and block_id in self.vace.vace_layers_mapping:
                x = x + vace_hints[self.vace.vace_layers_mapping[block_id]] * vace_scale
        return x

    def to_device(self, device):
        for name, module in self.named_children():
            if name == "blocks" and self.async_swap_flag:
                continue
            else:
                module.to(device, non_blocking=True)
        return self

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        cn_latents: Optional[torch.tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
        cond_flag: bool = True,
        add_condition: Optional[torch.tensor] = None,
        y_camera: Optional[torch.tensor] = None,
        vace_context: Optional[torch.tensor] = None,
        vace_scale: Optional[torch.tensor] = None,
        sparse_params: Optional[SparseAttentionParams] = None,
        freq_start_idx: int = 0,
        freq_interval: int = 1,
    ):
        with amp.autocast("cuda", dtype=torch.float32):
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        t = t.to(self.dtype)
        t_mod = t_mod.to(self.dtype)
        context = self.text_embedding(context)

        if self.has_image_input and clip_feature is not None:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)
        if add_condition is not None:
            x = add_condition + x  ####
        # print(f"[DEBUG] x.shape after patchify = {x.shape}")  # [B, N, C]
        if cn_latents is not None:
            condition = cn_latents
            condition = rearrange(condition, "b c f h w -> b (f h w) c").contiguous()
            # print("condition:",condition.shape)
            mean_x, std_x = (
                torch.mean(x, dim=(1, 2), keepdim=True),
                torch.std(x, dim=(1, 2), keepdim=True),
            )
            mean_condition, std_condition = (
                torch.mean(condition, dim=(1, 2), keepdim=True),
                torch.std(condition, dim=(1, 2), keepdim=True),
            )
            condition = (condition - mean_condition) * (
                std_x / (std_condition + 1e-5)
            ) + mean_x
            # print("x",x.shape,condition.shape)
            x = x + condition * 0.2
        # print(f"[DEBUG] x.shape after add = {x.shape}")  # [B, N, C]
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            y_camera = rearrange(y_camera, "b c f h w -> b (f h w) c").contiguous()
            x = x + y_camera

        freqs = (
            torch.cat(
                [
                    self.freqs[0][
                        freq_start_idx : freq_interval * f
                        + freq_start_idx : freq_interval
                    ]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        if getattr(self, "use_cfgp", False):
            # cfg rank0: positive prompt
            # cfg rank1: negative prompt
            cond_flag = False if get_cfg_rank() else True

        use_usp = getattr(self, "use_usp", False)
        vace_hints = None
        if vace_context is not None and self.vace is not None:
            vace_hints = self.vace(
                x, vace_context, context, t_mod, freqs, attn_impl=attn_impl
            )

        with sequence_parallel([x, freqs], seq_dims=[1, 0], enabled=use_usp):
            if self.cond_mag_cache is not None:
                mag_cache = self.cond_mag_cache if cond_flag else self.uncond_mag_cache
                mag_cache_update = mag_cache.check()
                if mag_cache_update:
                    x = mag_cache.update(x)
                else:
                    ori_x = x
                    x = self.forward_blocks(
                        x,
                        context,
                        t_mod,
                        freqs,
                        attn_impl,
                        vace_context,
                        vace_hints,
                        vace_scale,
                        sparse_params=sparse_params,
                    )
                    mag_cache.store(x, ori_x)

            elif self.cond_tea_cache is not None:
                input_x = x.clone()
                tea_cache = self.cond_tea_cache if cond_flag else self.uncond_tea_cache
                tea_cache_update = tea_cache.check(x, t_mod, use_usp=False)
                if tea_cache_update:
                    x = tea_cache.update(x)
                else:
                    x = self.forward_blocks(
                        x,
                        context,
                        t_mod,
                        freqs,
                        attn_impl,
                        vace_context,
                        vace_hints,
                        vace_scale,
                        sparse_params=sparse_params,
                    )
                    tea_cache.store(input_x, x, t_mod, t)
            else:
                x = self.forward_blocks(
                    x,
                    context,
                    t_mod,
                    freqs,
                    attn_impl,
                    vace_context,
                    vace_hints,
                    vace_scale,
                    sparse_params=sparse_params,
                )
            x = self.head(x, t)
            (x,) = sequence_parallel_unshard((x,), seq_dims=(1,), seq_lens=(f * h * w,))
        x = self.unpatchify(x, (f, h, w))
        return x

    def reset_tea_cache(self):
        self.cond_tea_cache = None
        self.uncond_tea_cache = None

    def reset_mag_cache(self):
        self.cond_mag_cache = None
        self.uncond_mag_cache = None

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    def get_tp_plan(self):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
        )
        from torch.distributed.tensor import Replicate, Shard

        tp_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_embedding.0": ColwiseParallel(),
            "time_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Shard(1), None, None, None),  # sequence parallel
                use_local_output=True,
            ),
            "head": PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        }
        for idx in range(len(self.blocks)):
            tp_plan.update(
                {
                    f"blocks.{idx}.self_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.self_attn.q": ColwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.self_attn.k": ColwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.self_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.self_attn.o": RowwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.self_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.self_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.cross_attn.q": ColwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.cross_attn.k": ColwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.cross_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.o": RowwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.cross_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.k_img": ColwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    f"blocks.{idx}.cross_attn.v_img": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.norm_k_img": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.ffn": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    f"blocks.{idx}.ffn.0": ColwiseParallel(),
                    f"blocks.{idx}.ffn.2": RowwiseParallel(output_layouts=Shard(1)),
                }
            )
        return tp_plan


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        i2v_rename_dict = {
            "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",  #  [1280],
            "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
            "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
            "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
            "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
            "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
            "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
            "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
            "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
            "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
            "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
            "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
            "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        }
        rename_dict.update(i2v_rename_dict)
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(
                        name_.split(".")[:1]
                        + [name.split(".")[1]]
                        + name_.split(".")[2:]
                    )
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "7cf3a086b49216bded0728ce78d59687":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("vace")
        }
        state_dict_hash = hash_state_dict_keys(state_dict)
        if state_dict_hash == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif state_dict_hash == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif state_dict_hash == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif state_dict_hash == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            # FL2V
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True,
            }
        # Wan Fun Camera Control
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif state_dict_hash == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif state_dict_hash == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "4cf556355bc7e9b6545b38f4930f60b1":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif state_dict_hash == "9d0240d8e7650a9ec65b2b617cc9c357":
            # Wan-AI/Wan2.2-T2V-A14B fp8
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        # 10B
        elif hash_state_dict_keys(state_dict) == "aa40ad461177ea9e288b34f0c103beac":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 30,
                "eps": 1e-6,
                "has_image_pos_emb": False,
            }
        else:
            config = {}
        return state_dict, config
