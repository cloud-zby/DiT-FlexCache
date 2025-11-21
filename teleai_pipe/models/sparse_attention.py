import torch
from typing import Optional, Literal

from dataclasses import dataclass
from typing import Optional, Dict, Any
from loguru import logger

try:
    from tele_fuser.radial_attn.attn import RadialAttention
    from tele_fuser.radial_attn.sparse_parallel import SparseLongContextAttention
    from tele_fuser.radial_attn.attn import pad_qkv
except ImportError:
    logger.warning("failed to import radial, please install TeleFuser")


@dataclass
class SparseAttentionParams:
    sparse_impl: Optional[str] = None

    # radial params
    mask_map: Optional[Dict] = None
    decay_factor: Optional[float] = None
    numeral_timestep: int = 0
    dense_timestep: int = 40
    layer_idx: int = 0
    dense_block: int = 40

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Invalid parameter: {k}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sparse_impl": self.sparse_impl,
            "mask_map": self.mask_map,
            "decay_factor": self.decay_factor,
            "numeral_timestep": self.numeral_timestep,
            "dense_timestep": self.dense_timestep,
            "dense_block": self.dense_block,
            "layer_idx": self.layer_idx,
        }


@torch.compiler.disable
def sparse_attention(
    q,
    k,
    v,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
    sparse_params: Optional[SparseAttentionParams] = None,
):
    # q = pad_qkv(q, 128)
    # k = pad_qkv(k, 128)
    # v = pad_qkv(v, 128)
    if sparse_params.sparse_impl == "radial":
        dense_flag = (
            sparse_params.numeral_timestep < sparse_params.dense_timestep
            or sparse_params.layer_idx < sparse_params.dense_block
        )
        if dense_flag:
            output = RadialAttention(
                query=q,
                key=k,
                value=v,
                mask_map=sparse_params.mask_map,
                sparsity_type="dense",
                block_size=128,
                decay_factor=sparse_params.decay_factor,
                model_type="wan",
                use_sage_attention=True,
            )
        else:
            output = RadialAttention(
                query=q,
                key=k,
                value=v,
                mask_map=sparse_params.mask_map,
                sparsity_type="radial",
                block_size=128,
                decay_factor=sparse_params.decay_factor,
                model_type="wan",
                use_sage_attention=True,
            )
        return output


@torch.compiler.disable
def sparse_long_context_attention(
    q,
    k,
    v,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
    sparse_params: Optional[SparseAttentionParams] = None,
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    if input_layout == "BNSD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    if sparse_params.sparse_impl == "radial":
        dense_flag = (
            sparse_params.numeral_timestep < sparse_params.dense_timestep
            or sparse_params.layer_idx < sparse_params.dense_block
        )
        attn_func = SparseLongContextAttention()
        if dense_flag:
            output = attn_func(
                query=q,
                key=k,
                value=v,
                mask_map=sparse_params.mask_map,
                sparsity_type="dense",
                block_size=128,
                decay_factor=sparse_params.decay_factor,
                model_type="wan",
                use_sage_attention=True,
            )
        else:
            output = attn_func(
                query=q,
                key=k,
                value=v,
                mask_map=sparse_params.mask_map,
                sparsity_type="radial",
                block_size=128,
                decay_factor=sparse_params.decay_factor,
                model_type="wan",
                use_sage_attention=True,
            )

    return output
