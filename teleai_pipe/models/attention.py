import torch
import torch.nn as nn
from loguru import logger
from einops import rearrange
from typing import Optional, Literal

try:
    from tele_fuser.yunchang import LongContextAttention
    from tele_fuser.yunchang.kernels import AttnType
except ImportError:
    logger.warning("failed to import yunchang, please install TeleFuser")
from teleai_pipe.utils.flag import (
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
    XFORMERS_AVAILABLE,
    SDPA_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    SPARGE_ATTN_AVAILABLE,
)

if FLASH_ATTN_3_AVAILABLE:
    from flash_attn_interface import flash_attn_func as flash_attn3
if FLASH_ATTN_2_AVAILABLE:
    from flash_attn import flash_attn_func as flash_attn2
if XFORMERS_AVAILABLE:
    import xformers.ops.memory_efficient_attention as xformers_attn
if SDPA_AVAILABLE:

    def sdpa_attn_cudnn(q, k, v, attn_mask=None, scale=None):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=scale
            )
        return out


if SAGE_ATTN_AVAILABLE:
    from tele_fuser.sageattention import (
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
    )


if SPARGE_ATTN_AVAILABLE:
    from tele_fuser.spas_sage_attn import spas_sage2_attn_meansim_cuda

    def sparge_attn(q, k, v, attn_mask=None, scale=None):
        out = spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=attn_mask, scale=scale)
        return out


def eager_attn(query, key, value, attn_mask=None, scale=None):
    scale = 1 / query.shape[-1] ** 0.5 if scale is None else scale
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(-1)
    return attn @ value


def attention(
    q,
    k,
    v,
    attn_mask=None,
    attn_impl: Literal[
        "sdpa",
        "cudnn",
        "eager",
        "flash_attn_2",
        "flash_attn_3",
        "xformers",
        "sage_attn_8_8",
        "sage_attn_8_16",
        "sage_attn_8_8_sm90",
        "sparge_attn",
    ] = "sdpa",
    scale: Optional[float] = None,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
):
    if attn_impl not in [
        "eager",
        "flash_attn_2",
        "flash_attn_3",
        "xformers",
        "sdpa",
        "cudnn",
        "sage_attn_8_8",
        "sage_attn_8_16",
        "sage_attn_8_8_sm90",
        "sparge_attn",
    ]:
        logger.error(f"attn impl {attn_impl} is not supported")
        exit()
    BNSD_LIST = [
        "eager",
        "cudnn",
        "sdpa",
        "sage_attn_8_8",
        "sage_attn_8_16",
        "sage_attn_8_8_sm90",
        "sparge_attn",
    ]
    if input_layout == "BSND" and attn_impl in BNSD_LIST:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if attn_impl == "eager":
        output = eager_attn(q, k, v, attn_mask=attn_mask, scale=scale)
    elif attn_impl == "flash_attn_3" and FLASH_ATTN_3_AVAILABLE:
        output = flash_attn3(q, k, v, softmax_scale=scale)
    elif attn_impl == "flash_attn_2" and FLASH_ATTN_2_AVAILABLE:
        output = flash_attn2(q, k, v, softmax_scale=scale)
    elif attn_impl == "xformers" and XFORMERS_AVAILABLE:
        output = xformers_attn(q, k, v, attn_bias=attn_mask, scale=scale)
    elif attn_impl == "cudnn" and SDPA_AVAILABLE:
        output = sdpa_attn_cudnn(q, k, v, attn_mask=attn_mask, scale=scale)
    elif attn_impl == "sdpa" and SDPA_AVAILABLE:
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale
        )
    elif attn_impl == "sage_attn_8_8" and SAGE_ATTN_AVAILABLE:
        output = sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, attn_mask=attn_mask, sm_scale=scale
        )
    elif attn_impl == "sage_attn_8_16" and SAGE_ATTN_AVAILABLE:
        output = sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, attn_mask=attn_mask, sm_scale=scale
        )
    elif attn_impl == "sage_attn_8_8_sm90" and SAGE_ATTN_AVAILABLE:
        output = sageattn_qk_int8_pv_fp8_cuda_sm90(
            q, k, v, attn_mask=attn_mask, sm_scale=scale
        )
    elif attn_impl == "sparge_attn" and SPARGE_ATTN_AVAILABLE:
        output = sparge_attn(q, k, v, attn_mask=attn_mask, scale=scale)
    else:
        logger.error(f"set attention impl {attn_impl} while it is not avaliable")
    if (output_layout == "BSND" and attn_impl in BNSD_LIST) or (
        output_layout == "BNSD" and attn_impl not in BNSD_LIST
    ):
        output = output.transpose(1, 2)

    return output


def long_context_attention(
    q,
    k,
    v,
    attn_impl: Optional[str] = "sdpa",
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    assert attn_impl in [
        "eager",
        "flash_attn_2",
        "flash_attn_3",
        "xformers",
        "sdpa",
        "sparge_attn",
        "sage_attn_8_8",
        "sage_attn_8_16",
        "sage_attn_8_8_sm90",
    ]
    if input_layout == "BNSD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if attn_impl == "flash_attn_3":
        attn_func = LongContextAttention(attn_type=AttnType.FA3, use_sync=True)
    elif attn_impl == "flash_attn_2":
        attn_func = LongContextAttention(attn_type=AttnType.FA, use_sync=True)
    elif attn_impl == "sdpa":
        attn_func = LongContextAttention(attn_type=AttnType.TORCH, use_sync=True)
    elif attn_impl == "sage_attn_8_16":
        attn_func = LongContextAttention(attn_type=AttnType.SAGE_FP16, use_sync=True)
    elif attn_impl == "sage_attn_8_8":
        attn_func = LongContextAttention(attn_type=AttnType.SAGE_FP8, use_sync=True)
    elif attn_impl == "sage_attn_8_8_sm90":
        attn_func = LongContextAttention(
            attn_type=AttnType.SAGE_FP8_SM90, use_sync=True
        )
    elif attn_impl == "sparge_attn":
        attn_func = LongContextAttention(attn_type=AttnType.SPARSE_SAGE, use_sync=True)
    else:
        raise ValueError(f"Invalid long context attention implementation: {attn_impl}")
    output = attn_func(q, k, v, softmax_scale=scale)
    if output_layout == "BNSD":
        output = output.transpose(1, 2)
    return output


class Attention(nn.Module):
    def __init__(
        self,
        q_dim,
        num_heads,
        head_dim,
        kv_dim=None,
        bias_q=False,
        bias_kv=False,
        bias_out=False,
        scale=None,
        attn_impl: str = "sdpa",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(q_dim, dim_inner, bias=bias_q, device=device, dtype=dtype)
        self.to_k = nn.Linear(
            kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype
        )
        self.to_v = nn.Linear(
            kv_dim, dim_inner, bias=bias_kv, device=device, dtype=dtype
        )
        self.to_out = nn.Linear(
            dim_inner, q_dim, bias=bias_out, device=device, dtype=dtype
        )
        self.attn_impl = attn_impl
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if y is None:
            y = x
        q = rearrange(self.to_q(x), "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(self.to_k(y), "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(self.to_v(y), "b s (n d) -> b s n d", n=self.num_heads)
        out = attention(
            q, k, v, attn_mask=attn_mask, attn_impl=self.attn_impl, scale=self.scale
        )
        out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
        return self.to_out(out)


class AttentionV1(torch.nn.Module):
    def __init__(
        self,
        q_dim,
        num_heads,
        head_dim,
        kv_dim=None,
        bias_q=False,
        bias_kv=False,
        bias_out=False,
    ):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def interact_with_ipadapter(self, hidden_states, q, ip_k, ip_v, scale=1.0):
        batch_size = q.shape[0]
        ip_k = ip_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_v = ip_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(
            q, ip_k, ip_v
        )
        hidden_states = hidden_states + scale * ip_hidden_states
        return hidden_states

    def torch_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attn_mask=None,
        ipadapter_kwargs=None,
        qkv_preprocessor=None,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if qkv_preprocessor is not None:
            q, k, v = qkv_preprocessor(q, k, v)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        if ipadapter_kwargs is not None:
            hidden_states = self.interact_with_ipadapter(
                hidden_states, q, **ipadapter_kwargs
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def xformers_forward(
        self, hidden_states, encoder_hidden_states=None, attn_mask=None
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b f (n d) -> (b n) f d", n=self.num_heads)
        k = rearrange(k, "b f (n d) -> (b n) f d", n=self.num_heads)
        v = rearrange(v, "b f (n d) -> (b n) f d", n=self.num_heads)

        if attn_mask is not None:
            hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        else:
            import xformers.ops as xops

            hidden_states = xops.memory_efficient_attention(q, k, v)
        hidden_states = rearrange(
            hidden_states, "(b n) f d -> b f (n d)", n=self.num_heads
        )

        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attn_mask=None,
        ipadapter_kwargs=None,
        qkv_preprocessor=None,
    ):
        return self.torch_forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_mask=attn_mask,
            ipadapter_kwargs=ipadapter_kwargs,
            qkv_preprocessor=qkv_preprocessor,
        )
