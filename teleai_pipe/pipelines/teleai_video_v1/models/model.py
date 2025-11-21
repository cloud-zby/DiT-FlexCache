import torch
from teleai_pipe.utils.utils import hash_state_dict_keys
from torch import nn
import torch.distributed as dist
from teleai_pipe.models.sd3_dit import TimestepEmbeddings, RMSNorm
from teleai_pipe.plugin.teacache import TeaCache
from teleai_pipe.utils.utils import init_weights_on_device
from einops import rearrange, repeat
from typing import Union, Tuple, List
import torch
import torch.nn as nn
from loguru import logger
from teleai_pipe.models.attention import attention as attn_func
from teleai_pipe.models.attention import long_context_attention as long_attn_func
from teleai_pipe.distributed.parallel import (
    get_sp_world_size,
    get_sp_rank,
    get_sp_group,
    get_cfg_world_size,
    get_cfg_rank,
    get_cfg_group,
)


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


def HunyuanVideoRope(latents):
    def _to_tuple(x, dim=2):
        if isinstance(x, int):
            return (x,) * dim
        elif len(x) == dim:
            return x
        else:
            raise ValueError(f"Expected length {dim} or int, but got {x}")

    def get_meshgrid_nd(start, *args, dim=2):
        """
        Get n-D meshgrid with start, stop and num.

        Args:
            start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
                step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
                should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
                n-tuples.
            *args: See above.
            dim (int): Dimension of the meshgrid. Defaults to 2.

        Returns:
            grid (np.ndarray): [dim, ...]
        """
        if len(args) == 0:
            # start is grid_size
            num = _to_tuple(start, dim=dim)
            start = (0,) * dim
            stop = num
        elif len(args) == 1:
            # start is start, args[0] is stop, step is 1
            start = _to_tuple(start, dim=dim)
            stop = _to_tuple(args[0], dim=dim)
            num = [stop[i] - start[i] for i in range(dim)]
        elif len(args) == 2:
            # start is start, args[0] is stop, args[1] is num
            start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
            stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
            num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
        else:
            raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

        # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
        axis_grid = []
        for i in range(dim):
            a, b, n = start[i], stop[i], num[i]
            g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
            axis_grid.append(g)
        grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
        grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

        return grid

    def get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[torch.FloatTensor, int],
        theta: float = 10000.0,
        use_real: bool = False,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Precompute the frequency tensor for complex exponential (cis) with given dimensions.
        (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

        This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
        and the end index 'end'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (bool, optional): If True, return real part and imaginary part separately.
                                    Otherwise, return complex numbers.
            theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

        Returns:
            freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
            freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
        """
        if isinstance(pos, int):
            pos = torch.arange(pos).float()

        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        if theta_rescale_factor != 1.0:
            theta *= theta_rescale_factor ** (dim / (dim - 2))

        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )  # [D/2]
        # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
        freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
            return freqs_cos, freqs_sin
        else:
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # complex64     # [S, D/2]
            return freqs_cis

    def get_nd_rotary_pos_embed(
        rope_dim_list,
        start,
        *args,
        theta=10000.0,
        use_real=False,
        theta_rescale_factor: Union[float, List[float]] = 1.0,
        interpolation_factor: Union[float, List[float]] = 1.0,
    ):
        """
        This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

        Args:
            rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
                sum(rope_dim_list) should equal to head_dim of attention layer.
            start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
                args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
            *args: See above.
            theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
                Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
                part and an imaginary part separately.
            theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

        Returns:
            pos_embed (torch.Tensor): [HW, D/2]
        """

        grid = get_meshgrid_nd(
            start, *args, dim=len(rope_dim_list)
        )  # [3, W, H, D] / [2, W, H]

        if isinstance(theta_rescale_factor, int) or isinstance(
            theta_rescale_factor, float
        ):
            theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
        elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
            theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
        assert len(theta_rescale_factor) == len(
            rope_dim_list
        ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

        if isinstance(interpolation_factor, int) or isinstance(
            interpolation_factor, float
        ):
            interpolation_factor = [interpolation_factor] * len(rope_dim_list)
        elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
            interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
        assert len(interpolation_factor) == len(
            rope_dim_list
        ), "len(interpolation_factor) should equal to len(rope_dim_list)"

        # use 1/ndim of dimensions to encode grid_axis
        embs = []
        for i in range(len(rope_dim_list)):
            emb = get_1d_rotary_pos_embed(
                rope_dim_list[i],
                grid[i].reshape(-1),
                theta,
                use_real=use_real,
                theta_rescale_factor=theta_rescale_factor[i],
                interpolation_factor=interpolation_factor[i],
            )  # 2 x [WHD, rope_dim_list[i]]
            embs.append(emb)

        if use_real:
            cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
            sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
            return cos, sin
        else:
            emb = torch.cat(embs, dim=1)  # (WHD, D/2)
            return emb

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        [16, 56, 56],
        [latents.shape[2], latents.shape[3] // 2, latents.shape[4] // 2],
        theta=256,
        use_real=True,
        theta_rescale_factor=1,
    )
    return freqs_cos, freqs_sin


class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_channels=16, embed_dim=3072):
        super().__init__()
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class IndividualTokenRefinerBlock(torch.nn.Module):
    def __init__(self, hidden_size=3072, num_heads=24):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.self_attn_proj = torch.nn.Linear(hidden_size, hidden_size)

        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                hidden_size, hidden_size * 2, device="cuda", dtype=torch.bfloat16
            ),
        )

    def forward(self, x, c, attn_mask=None, attn_impl: str = "sdpa"):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        attn = attn_func(
            q,
            k,
            v,
            attn_impl=attn_impl,
            input_layout="BNSD",
            output_layout="BNSD",
            attn_mask=attn_mask,
        )
        attn = rearrange(attn, "B H L D -> B L (H D)")

        x = x + self.self_attn_proj(attn) * gate_msa.unsqueeze(1)
        x = x + self.mlp(self.norm2(x)) * gate_mlp.unsqueeze(1)

        return x


class SingleTokenRefiner(torch.nn.Module):
    def __init__(self, in_channels=4096, hidden_size=3072, depth=2):
        super().__init__()
        self.input_embedder = torch.nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbeddings(256, hidden_size, computation_device="cpu")
        self.c_embedder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = torch.nn.ModuleList(
            [IndividualTokenRefinerBlock(hidden_size=hidden_size) for _ in range(depth)]
        )

    def forward(self, x, t, mask=None, attn_impl: str = "sdpa"):
        timestep_aware_representations = self.t_embedder(t, dtype=torch.float32)

        mask_float = mask.float().unsqueeze(-1)
        context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        )
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        mask = mask.to(device=x.device, dtype=torch.bool)
        mask = repeat(mask, "B L -> B 1 D L", D=mask.shape[-1])
        mask = mask & mask.transpose(2, 3)
        mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, mask, attn_impl)

        return x


class ModulateDiT(torch.nn.Module):
    def __init__(self, hidden_size, factor=6):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.linear = torch.nn.Linear(hidden_size, factor * hidden_size)

    def forward(self, x):
        return self.linear(self.act(x))


def modulate(x, shift=None, scale=None):
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def reshape_for_broadcast(
    freqs_cis,
    x: torch.Tensor,
    head_first=False,
):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis,
    head_first: bool = False,
):
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class MMDoubleStreamBlockComponent(torch.nn.Module):
    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.heads_num = heads_num

        self.mod = ModulateDiT(hidden_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.to_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.norm_k = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.to_out = torch.nn.Linear(hidden_size, hidden_size)

        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * mlp_width_ratio),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_size * mlp_width_ratio, hidden_size),
        )

    def forward(self, hidden_states, conditioning, freqs_cis=None):
        mod1_shift, mod1_scale, mod1_gate, mod2_shift, mod2_scale, mod2_gate = self.mod(
            conditioning
        ).chunk(6, dim=-1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states, shift=mod1_shift, scale=mod1_scale
        )
        qkv = self.to_qkv(norm_hidden_states)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False)

        return (q, k, v), (mod1_gate, mod2_shift, mod2_scale, mod2_gate)

    def process_ff(self, hidden_states, attn_output, mod):
        mod1_gate, mod2_shift, mod2_scale, mod2_gate = mod
        hidden_states = hidden_states + self.to_out(attn_output) * mod1_gate.unsqueeze(
            1
        )
        hidden_states = hidden_states + self.ff(
            modulate(self.norm2(hidden_states), shift=mod2_shift, scale=mod2_scale)
        ) * mod2_gate.unsqueeze(1)
        return hidden_states


class MMDoubleStreamBlock(torch.nn.Module):
    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.component_a = MMDoubleStreamBlockComponent(
            hidden_size, heads_num, mlp_width_ratio
        )
        self.component_b = MMDoubleStreamBlockComponent(
            hidden_size, heads_num, mlp_width_ratio
        )

    def forward(
        self,
        hidden_states_a,
        hidden_states_b,
        conditioning,
        freqs_cis,
        attn_impl: str = "sdpa",
    ):
        (q_a, k_a, v_a), mod_a = self.component_a(
            hidden_states_a, conditioning, freqs_cis
        )
        (q_b, k_b, v_b), mod_b = self.component_b(
            hidden_states_b, conditioning, freqs_cis=None
        )

        q = torch.concat([q_a, q_b], dim=1).contiguous()
        k = torch.concat([k_a, k_b], dim=1).contiguous()
        v = torch.concat([v_a, v_b], dim=1).contiguous()
        if getattr(self, "use_usp", False):
            attn_output = long_attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BSND", output_layout="BSND"
            ).flatten(2, 3)
        else:
            attn_output = attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BSND", output_layout="BSND"
            ).flatten(2, 3)
        attn_output_a, attn_output_b = (
            attn_output[:, : -hidden_states_b.size(1)].contiguous(),
            attn_output[:, -hidden_states_b.size(1) :].contiguous(),
        )
        hidden_states_a = self.component_a.process_ff(
            hidden_states_a, attn_output_a, mod_a
        )
        hidden_states_b = self.component_b.process_ff(
            hidden_states_b, attn_output_b, mod_b
        )
        return hidden_states_a, hidden_states_b


class MMSingleStreamBlock(torch.nn.Module):
    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.heads_num = heads_num

        self.mod = ModulateDiT(hidden_size, factor=3)
        self.norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.to_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.norm_k = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.to_out = torch.nn.Linear(hidden_size, hidden_size)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * mlp_width_ratio),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_size * mlp_width_ratio, hidden_size, bias=False),
        )

    def forward(
        self,
        hidden_states,
        conditioning,
        freqs_cis=None,
        txt_len=256,
        attn_impl: str = "sdpa",
    ):
        mod_shift, mod_scale, mod_gate = self.mod(conditioning).chunk(3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states, shift=mod_shift, scale=mod_scale
        )
        qkv = self.to_qkv(norm_hidden_states)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q_a, q_b = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        k_a, k_b = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        q_a, k_a = apply_rotary_emb(q_a, k_a, freqs_cis, head_first=False)

        q = torch.concat([q_a, q_b], dim=1).contiguous()
        k = torch.concat([k_a, k_b], dim=1).contiguous()
        if getattr(self, "use_usp", False):
            attn_output = long_attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BSND", output_layout="BSND"
            ).flatten(2, 3)
        else:
            attn_output = attn_func(
                q, k, v, attn_impl=attn_impl, input_layout="BSND", output_layout="BSND"
            ).flatten(2, 3)
        hidden_states = hidden_states + self.to_out(attn_output) * mod_gate.unsqueeze(1)
        tmp = self.ff(norm_hidden_states)
        hidden_states = hidden_states + tmp * mod_gate.unsqueeze(1)
        return hidden_states


class TeleAIVideoDiTV1(torch.nn.Module):
    def __init__(
        self,
        in_channels=16,
        hidden_size=3072,
        text_dim=4096,
        num_double_blocks=20,
        num_single_blocks=40,
    ):
        super().__init__()
        self.img_in = PatchEmbed(in_channels=in_channels, embed_dim=hidden_size)
        self.txt_in = SingleTokenRefiner(in_channels=text_dim, hidden_size=hidden_size)
        self.time_in = TimestepEmbeddings(256, hidden_size, computation_device="cuda")
        self.vector_in = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.guidance_in = TimestepEmbeddings(
            256, hidden_size, computation_device="cuda"
        )
        self.double_blocks = torch.nn.ModuleList(
            [MMDoubleStreamBlock(hidden_size) for _ in range(num_double_blocks)]
        )
        self.single_blocks = torch.nn.ModuleList(
            [MMSingleStreamBlock(hidden_size) for _ in range(num_single_blocks)]
        )
        self.norm_out = AdaLayerNormContinuous(
            hidden_size, hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(hidden_size, 1 * 2 * 2 * 16)
        self.modulate_inp_layer = None

        # TODO: remove these parameters
        self.dtype = torch.bfloat16
        self.patch_size = [1, 2, 2]
        self.hidden_size = 3072
        self.heads_num = 24
        self.rope_dim_list = [16, 56, 56]
        self.cond_tea_cache = None
        self.uncond_tea_cache = None

    def modulate_input(self, img, vec):
        img_mod1_shift, img_mod1_scale, _, _, _, _ = self.modulate_inp_layer.mod(
            vec
        ).chunk(6, dim=-1)
        normed_inp = self.modulate_inp_layer.norm1(img)
        modulated_inp = normed_inp * (
            1 + img_mod1_scale.unsqueeze(1)
        ) + img_mod1_shift.unsqueeze(1)
        return modulated_inp

    def enable_cfgp(self):
        setattr(self, "use_cfgp", True)

    def enable_usp(self):
        setattr(self, "use_usp", True)
        for block in self.single_blocks:
            setattr(block, "use_usp", True)
        for block in self.double_blocks:
            setattr(block, "use_usp", True)

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

    def unpatchify(self, x, T, H, W):
        x = rearrange(
            x,
            "B (T H W) (C pT pH pW) -> B C (T pT) (H pH) (W pW)",
            H=H,
            W=W,
            pT=1,
            pH=2,
            pW=2,
        )
        return x

    def enable_block_wise_offload(self, warm_device="cuda", cold_device="cpu"):
        self.warm_device = warm_device
        self.cold_device = cold_device
        self.to(self.cold_device)

    def load_models_to_device(self, loadmodel_names=[], device="cpu"):
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to(device)
        torch.cuda.empty_cache()

    def prepare_freqs(self, latents):
        return HunyuanVideoRope(latents)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prompt_emb: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        pooled_prompt_emb: torch.Tensor = None,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        guidance: torch.Tensor = None,
        attn_impl: str = "sdpa",
        cond_flag: bool = True,
    ):
        B, C, T, H, W = x.shape
        vec = (
            self.time_in(t, dtype=torch.float32)
            + self.vector_in(pooled_prompt_emb)
            + self.guidance_in(guidance * 1000, dtype=torch.float32)
        )
        img = self.img_in(x)
        use_usp = getattr(self, "use_usp", False)
        if getattr(self, "use_cfgp", False):
            # cfg rank0: positive prompt
            # cfg rank1: negative prompt
            cond_flag = False if get_cfg_rank() else True

        if use_usp:
            s, p = img.size(1), get_sp_world_size()  # (sequence_length, parallelism)
            split_size = [s // p + 1 if i < s % p else s // p for i in range(p)]
            img = torch.split(img, split_size, dim=1)[get_sp_rank()]
            freqs_cos = torch.split(freqs_cos, split_size, dim=0)[get_sp_rank()]
            freqs_sin = torch.split(freqs_sin, split_size, dim=0)[get_sp_rank()]
        txt = self.txt_in(prompt_emb, t, text_mask, "cudnn")
        valid_text_len = text_mask.sum(dim=1, dtype=torch.int)
        txt = txt[:, :valid_text_len, :]
        if self.cond_tea_cache is None and self.uncond_tea_cache is None:
            for block in self.double_blocks:
                img, txt = block(img, txt, vec, (freqs_cos, freqs_sin), attn_impl)
            x = torch.concat([img, txt], dim=1)
            for block in self.single_blocks:
                x = block(x, vec, (freqs_cos, freqs_sin), valid_text_len, attn_impl)
            img = x[:, :-valid_text_len]
        else:
            tea_cache = self.cond_tea_cache if cond_flag else self.uncond_tea_cache
            # get modulated_inp
            modulated_inp = self.modulate_input(img, vec)
            tea_cache_update = tea_cache.check(img, modulated_inp, use_usp=use_usp)
            hidden_states = img.clone()
            if tea_cache_update:
                img = tea_cache.update(img)
            else:
                for block in self.double_blocks:
                    img, txt = block(img, txt, vec, (freqs_cos, freqs_sin), attn_impl)
                x = torch.concat([img, txt], dim=1)
                for block in self.single_blocks:
                    x = block(x, vec, (freqs_cos, freqs_sin), valid_text_len, attn_impl)
                img = x[:, :-valid_text_len]
                tea_cache.store(
                    hidden_states,
                    img,
                    modulated_inp,
                    self.time_in(t, dtype=torch.float32),
                )
        img = self.norm_out(img, vec)
        img = self.proj_out(img)
        if use_usp:
            b, d = img.size(0), img.size(2)  # (batch_size, out_dim)
            xs = [
                torch.zeros((b, s, d), dtype=img.dtype, device=img.device)
                for s in split_size
            ]
            dist.all_gather(xs, img, group=get_sp_group())
            img = torch.concat(xs, dim=1)
        img = self.unpatchify(img, T=T // 1, H=H // 2, W=W // 2)
        return img

    def enable_auto_offload(self, dtype=torch.bfloat16, device="cuda"):
        def cast_to(weight, dtype=None, device=None, copy=False):
            if device is None or weight.device == device:
                if not copy:
                    if dtype is None or weight.dtype == dtype:
                        return weight
                return weight.to(dtype=dtype, copy=copy)

            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight)
            return r

        def cast_weight(s, input=None, dtype=None, device=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            return weight

        def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if bias_dtype is None:
                    bias_dtype = dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            bias = cast_to(s.bias, bias_dtype, device) if s.bias is not None else None
            return weight, bias

        class quantized_layer:
            class Linear(torch.nn.Linear):
                def __init__(
                    self, *args, dtype=torch.bfloat16, device="cuda", **kwargs
                ):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device

                def block_forward_(self, x, i, j, dtype, device):
                    weight_ = cast_to(
                        self.weight[
                            j * self.block_size : (j + 1) * self.block_size,
                            i * self.block_size : (i + 1) * self.block_size,
                        ],
                        dtype=dtype,
                        device=device,
                    )
                    if self.bias is None or i > 0:
                        bias_ = None
                    else:
                        bias_ = cast_to(
                            self.bias[j * self.block_size : (j + 1) * self.block_size],
                            dtype=dtype,
                            device=device,
                        )
                    x_ = x[..., i * self.block_size : (i + 1) * self.block_size]
                    y_ = torch.nn.functional.linear(x_, weight_, bias_)
                    del x_, weight_, bias_
                    torch.cuda.empty_cache()
                    return y_

                def block_forward(self, x, **kwargs):
                    # This feature can only reduce 2GB VRAM, so we disable it.
                    y = torch.zeros(
                        x.shape[:-1] + (self.out_features,),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    for i in range(
                        (self.in_features + self.block_size - 1) // self.block_size
                    ):
                        for j in range(
                            (self.out_features + self.block_size - 1) // self.block_size
                        ):
                            y[..., j * self.block_size : (j + 1) * self.block_size] += (
                                self.block_forward_(
                                    x, i, j, dtype=x.dtype, device=x.device
                                )
                            )
                    return y

                def forward(self, x, **kwargs):
                    weight, bias = cast_bias_weight(
                        self, x, dtype=self.dtype, device=self.device
                    )
                    return torch.nn.functional.linear(x, weight, bias)

            class RMSNorm(torch.nn.Module):
                def __init__(self, module, dtype=torch.bfloat16, device="cuda"):
                    super().__init__()
                    self.module = module
                    self.dtype = dtype
                    self.device = device

                def forward(self, hidden_states, **kwargs):
                    input_dtype = hidden_states.dtype
                    variance = (
                        hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
                    )
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + self.module.eps
                    )
                    hidden_states = hidden_states.to(input_dtype)
                    if self.module.weight is not None:
                        weight = cast_weight(
                            self.module,
                            hidden_states,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        hidden_states = hidden_states * weight
                    return hidden_states

            class Conv3d(torch.nn.Conv3d):
                def __init__(
                    self, *args, dtype=torch.bfloat16, device="cuda", **kwargs
                ):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device

                def forward(self, x):
                    weight, bias = cast_bias_weight(
                        self, x, dtype=self.dtype, device=self.device
                    )
                    return torch.nn.functional.conv3d(
                        x,
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )

            class LayerNorm(torch.nn.LayerNorm):
                def __init__(
                    self, *args, dtype=torch.bfloat16, device="cuda", **kwargs
                ):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device

                def forward(self, x):
                    if self.weight is not None and self.bias is not None:
                        weight, bias = cast_bias_weight(
                            self, x, dtype=self.dtype, device=self.device
                        )
                        return torch.nn.functional.layer_norm(
                            x, self.normalized_shape, weight, bias, self.eps
                        )
                    else:
                        return torch.nn.functional.layer_norm(
                            x, self.normalized_shape, self.weight, self.bias, self.eps
                        )

        def replace_layer(model, dtype=torch.bfloat16, device="cuda"):
            for name, module in model.named_children():
                if isinstance(module, torch.nn.Linear):
                    with init_weights_on_device():
                        new_layer = quantized_layer.Linear(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            dtype=dtype,
                            device=device,
                        )
                    new_layer.load_state_dict(module.state_dict(), assign=True)
                    setattr(model, name, new_layer)
                elif isinstance(module, torch.nn.Conv3d):
                    with init_weights_on_device():
                        new_layer = quantized_layer.Conv3d(
                            module.in_channels,
                            module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            dtype=dtype,
                            device=device,
                        )
                    new_layer.load_state_dict(module.state_dict(), assign=True)
                    setattr(model, name, new_layer)
                elif isinstance(module, RMSNorm):
                    new_layer = quantized_layer.RMSNorm(
                        module, dtype=dtype, device=device
                    )
                    setattr(model, name, new_layer)
                elif isinstance(module, torch.nn.LayerNorm):
                    with init_weights_on_device():
                        new_layer = quantized_layer.LayerNorm(
                            module.normalized_shape,
                            elementwise_affine=module.elementwise_affine,
                            eps=module.eps,
                            dtype=dtype,
                            device=device,
                        )
                    new_layer.load_state_dict(module.state_dict(), assign=True)
                    setattr(model, name, new_layer)
                else:
                    replace_layer(module, dtype=dtype, device=device)

        replace_layer(self, dtype=dtype, device=device)

    @staticmethod
    def state_dict_converter():
        return TeleAIVideoDiTStateDictConverter()


class TeleAIVideoDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        src_model_dict = state_dict
        target_model_dict = {}

        def common_convert(
            src_model, dst_model, convert_map, src_replace=None, dst_replace=None
        ):
            for k, v in convert_map.items():
                if dst_replace is not None:
                    k = k.replace(dst_replace[0], dst_replace[1])
                if isinstance(v, list):
                    if src_replace is not None:
                        v = [vv.replace(src_replace[0], src_replace[1]) for vv in v]
                    src_param_list = [src_model[vv] for vv in v]
                    src_param = torch.concat(src_param_list, dim=0)
                elif isinstance(v, tuple):
                    v = list(v)
                    if src_replace is not None:
                        v[0] = v[0].replace(src_replace[0], src_replace[1])
                    src_param = src_model[v[0]][:, v[1][0] : v[1][1]]
                else:
                    if src_replace is not None:
                        v = v.replace(src_replace[0], src_replace[1])
                    src_param = src_model[v]
                dst_model[k] = src_param

        img_in_map = {
            "img_in.proj.weight": "x_embedder.proj.weight",
            "img_in.proj.bias": "x_embedder.proj.bias",
        }
        time_in_map = {
            "time_in.timestep_embedder.0.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "time_in.timestep_embedder.0.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "time_in.timestep_embedder.2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
            "time_in.timestep_embedder.2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        }
        vector_in_map = {
            "vector_in.0.weight": "time_text_embed.text_embedder.linear_1.weight",
            "vector_in.0.bias": "time_text_embed.text_embedder.linear_1.bias",
            "vector_in.2.weight": "time_text_embed.text_embedder.linear_2.weight",
            "vector_in.2.bias": "time_text_embed.text_embedder.linear_2.bias",
        }
        guidance_in_map = {
            "guidance_in.timestep_embedder.0.weight": "time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.timestep_embedder.0.bias": "time_text_embed.guidance_embedder.linear_1.bias",
            "guidance_in.timestep_embedder.2.weight": "time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.timestep_embedder.2.bias": "time_text_embed.guidance_embedder.linear_2.bias",
        }
        final_layer_map = {
            "final_layer.linear.weight": "proj_out.weight",
            "final_layer.linear.bias": "proj_out.bias",
            "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
            "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        }
        txt_in_map = {
            "txt_in.input_embedder.weight": "context_embedder.proj_in.weight",
            "txt_in.input_embedder.bias": "context_embedder.proj_in.bias",
            "txt_in.t_embedder.timestep_embedder.0.weight": "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
            "txt_in.t_embedder.timestep_embedder.0.bias": "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
            "txt_in.t_embedder.timestep_embedder.2.weight": "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
            "txt_in.t_embedder.timestep_embedder.2.bias": "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
            "txt_in.c_embedder.0.weight": "context_embedder.time_text_embed.text_embedder.linear_1.weight",
            "txt_in.c_embedder.0.bias": "context_embedder.time_text_embed.text_embedder.linear_1.bias",
            "txt_in.c_embedder.2.weight": "context_embedder.time_text_embed.text_embedder.linear_2.weight",
            "txt_in.c_embedder.2.bias": "context_embedder.time_text_embed.text_embedder.linear_2.bias",
            "txt_in.blocks.0.norm1.weight": "context_embedder.token_refiner.refiner_blocks.0.norm1.weight",
            "txt_in.blocks.0.norm1.bias": "context_embedder.token_refiner.refiner_blocks.0.norm1.bias",
            "txt_in.blocks.0.self_attn_qkv.weight": [
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.weight",
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_k.weight",
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_v.weight",
            ],
            "txt_in.blocks.0.self_attn_qkv.bias": [
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.bias",
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_k.bias",
                "context_embedder.token_refiner.refiner_blocks.0.attn.to_v.bias",
            ],
            "txt_in.blocks.0.self_attn_proj.weight": "context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0.weight",
            "txt_in.blocks.0.self_attn_proj.bias": "context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0.bias",
            "txt_in.blocks.0.norm2.weight": "context_embedder.token_refiner.refiner_blocks.0.norm2.weight",
            "txt_in.blocks.0.norm2.bias": "context_embedder.token_refiner.refiner_blocks.0.norm2.bias",
            "txt_in.blocks.0.mlp.0.weight": "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.weight",
            "txt_in.blocks.0.mlp.0.bias": "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.bias",
            "txt_in.blocks.0.mlp.2.weight": "context_embedder.token_refiner.refiner_blocks.0.ff.net.2.weight",
            "txt_in.blocks.0.mlp.2.bias": "context_embedder.token_refiner.refiner_blocks.0.ff.net.2.bias",
            "txt_in.blocks.0.adaLN_modulation.1.weight": "context_embedder.token_refiner.refiner_blocks.0.norm_out.linear.weight",
            "txt_in.blocks.0.adaLN_modulation.1.bias": "context_embedder.token_refiner.refiner_blocks.0.norm_out.linear.bias",
            "txt_in.blocks.1.norm1.weight": "context_embedder.token_refiner.refiner_blocks.1.norm1.weight",
            "txt_in.blocks.1.norm1.bias": "context_embedder.token_refiner.refiner_blocks.1.norm1.bias",
            "txt_in.blocks.1.self_attn_qkv.weight": [
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_q.weight",
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_k.weight",
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_v.weight",
            ],
            "txt_in.blocks.1.self_attn_qkv.bias": [
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_q.bias",
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_k.bias",
                "context_embedder.token_refiner.refiner_blocks.1.attn.to_v.bias",
            ],
            "txt_in.blocks.1.self_attn_proj.weight": "context_embedder.token_refiner.refiner_blocks.1.attn.to_out.0.weight",
            "txt_in.blocks.1.self_attn_proj.bias": "context_embedder.token_refiner.refiner_blocks.1.attn.to_out.0.bias",
            "txt_in.blocks.1.norm2.weight": "context_embedder.token_refiner.refiner_blocks.1.norm2.weight",
            "txt_in.blocks.1.norm2.bias": "context_embedder.token_refiner.refiner_blocks.1.norm2.bias",
            "txt_in.blocks.1.mlp.0.weight": "context_embedder.token_refiner.refiner_blocks.1.ff.net.0.proj.weight",
            "txt_in.blocks.1.mlp.0.bias": "context_embedder.token_refiner.refiner_blocks.1.ff.net.0.proj.bias",
            "txt_in.blocks.1.mlp.2.weight": "context_embedder.token_refiner.refiner_blocks.1.ff.net.2.weight",
            "txt_in.blocks.1.mlp.2.bias": "context_embedder.token_refiner.refiner_blocks.1.ff.net.2.bias",
            "txt_in.blocks.1.adaLN_modulation.1.weight": "context_embedder.token_refiner.refiner_blocks.1.norm_out.linear.weight",
            "txt_in.blocks.1.adaLN_modulation.1.bias": "context_embedder.token_refiner.refiner_blocks.1.norm_out.linear.bias",
        }
        single_block_map = {
            "single_blocks.0.mod.linear.weight": "single_transformer_blocks.0.norm.linear.weight",
            "single_blocks.0.mod.linear.bias": "single_transformer_blocks.0.norm.linear.bias",
            "single_blocks.0.to_qkv.weight": [
                "single_transformer_blocks.0.attn.to_q.weight",
                "single_transformer_blocks.0.attn.to_k.weight",
                "single_transformer_blocks.0.attn.to_v.weight",
            ],
            "single_blocks.0.to_qkv.bias": [
                "single_transformer_blocks.0.attn.to_q.bias",
                "single_transformer_blocks.0.attn.to_k.bias",
                "single_transformer_blocks.0.attn.to_v.bias",
            ],
            "single_blocks.0.norm_q.weight": "single_transformer_blocks.0.attn.norm_q.weight",
            "single_blocks.0.norm_k.weight": "single_transformer_blocks.0.attn.norm_k.weight",
            "single_blocks.0.to_out.weight": (
                "single_transformer_blocks.0.proj_out.weight",
                [0, 3072],
            ),
            "single_blocks.0.to_out.bias": "single_transformer_blocks.0.proj_out.bias",
            "single_blocks.0.ff.0.weight": "single_transformer_blocks.0.proj_mlp.weight",
            "single_blocks.0.ff.0.bias": "single_transformer_blocks.0.proj_mlp.bias",
            "single_blocks.0.ff.2.weight": (
                "single_transformer_blocks.0.proj_out.weight",
                [3072, 3072 * 5],
            ),
        }
        double_block_map = {
            "double_blocks.0.component_a.mod.linear.weight": "transformer_blocks.0.norm1.linear.weight",
            "double_blocks.0.component_a.mod.linear.bias": "transformer_blocks.0.norm1.linear.bias",
            "double_blocks.0.component_a.to_qkv.weight": [
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.0.attn.to_k.weight",
                "transformer_blocks.0.attn.to_v.weight",
            ],
            "double_blocks.0.component_a.to_qkv.bias": [
                "transformer_blocks.0.attn.to_q.bias",
                "transformer_blocks.0.attn.to_k.bias",
                "transformer_blocks.0.attn.to_v.bias",
            ],
            "double_blocks.0.component_a.norm_q.weight": "transformer_blocks.0.attn.norm_q.weight",
            "double_blocks.0.component_a.norm_k.weight": "transformer_blocks.0.attn.norm_k.weight",
            "double_blocks.0.component_a.to_out.weight": "transformer_blocks.0.attn.to_out.0.weight",
            "double_blocks.0.component_a.to_out.bias": "transformer_blocks.0.attn.to_out.0.bias",
            "double_blocks.0.component_a.ff.0.weight": "transformer_blocks.0.ff.net.0.proj.weight",
            "double_blocks.0.component_a.ff.0.bias": "transformer_blocks.0.ff.net.0.proj.bias",
            "double_blocks.0.component_a.ff.2.weight": "transformer_blocks.0.ff.net.2.weight",
            "double_blocks.0.component_a.ff.2.bias": "transformer_blocks.0.ff.net.2.bias",
            "double_blocks.0.component_b.mod.linear.weight": "transformer_blocks.0.norm1_context.linear.weight",
            "double_blocks.0.component_b.mod.linear.bias": "transformer_blocks.0.norm1_context.linear.bias",
            "double_blocks.0.component_b.to_qkv.weight": [
                "transformer_blocks.0.attn.add_q_proj.weight",
                "transformer_blocks.0.attn.add_k_proj.weight",
                "transformer_blocks.0.attn.add_v_proj.weight",
            ],
            "double_blocks.0.component_b.to_qkv.bias": [
                "transformer_blocks.0.attn.add_q_proj.bias",
                "transformer_blocks.0.attn.add_k_proj.bias",
                "transformer_blocks.0.attn.add_v_proj.bias",
            ],
            "double_blocks.0.component_b.norm_q.weight": "transformer_blocks.0.attn.norm_added_q.weight",
            "double_blocks.0.component_b.norm_k.weight": "transformer_blocks.0.attn.norm_added_k.weight",
            "double_blocks.0.component_b.to_out.weight": "transformer_blocks.0.attn.to_add_out.weight",
            "double_blocks.0.component_b.to_out.bias": "transformer_blocks.0.attn.to_add_out.bias",
            "double_blocks.0.component_b.ff.0.weight": "transformer_blocks.0.ff_context.net.0.proj.weight",
            "double_blocks.0.component_b.ff.0.bias": "transformer_blocks.0.ff_context.net.0.proj.bias",
            "double_blocks.0.component_b.ff.2.weight": "transformer_blocks.0.ff_context.net.2.weight",
            "double_blocks.0.component_b.ff.2.bias": "transformer_blocks.0.ff_context.net.2.bias",
        }

        common_convert(src_model_dict, target_model_dict, img_in_map)
        common_convert(src_model_dict, target_model_dict, time_in_map)
        common_convert(src_model_dict, target_model_dict, vector_in_map)
        common_convert(src_model_dict, target_model_dict, guidance_in_map)
        common_convert(src_model_dict, target_model_dict, txt_in_map)
        for i in range(40):
            src_replace = (
                "single_transformer_blocks.0",
                f"single_transformer_blocks.{i}",
            )
            dst_replace = ("single_blocks.0", f"single_blocks.{i}")
            common_convert(
                src_model_dict,
                target_model_dict,
                single_block_map,
                src_replace,
                dst_replace,
            )
        for i in range(20):
            src_replace = ("transformer_blocks.0", f"transformer_blocks.{i}")
            dst_replace = ("double_blocks.0", f"double_blocks.{i}")
            common_convert(
                src_model_dict,
                target_model_dict,
                double_block_map,
                src_replace,
                dst_replace,
            )
        for k in final_layer_map.values():
            target_model_dict[k] = src_model_dict[k]
        model_cfg = dict()
        model_hash = hash_state_dict_keys(src_model_dict)
        if model_hash == "d8e20c19653a4ff8f0a3ceda39f38be7":
            model_cfg = dict(in_channels=16)
        elif model_hash == "9f048b1c44a21b23a27c5eb35cc5040f":
            model_cfg = dict(in_channels=33)
        elif model_hash == "163bc568e8fb1f994651339fbd713015":
            model_cfg = dict(in_channels=36)
        else:
            logger.error(f"model with hash {model_hash} is not supported")
            exit()
        return target_model_dict, model_cfg
