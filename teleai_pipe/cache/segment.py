from __future__ import annotations

from typing import List, Optional
import os


def segment_video(frames: List[object], object_prompt: str) -> List[object]:  # pragma: no cover - 占位
    """可选启用 LangSAM 分割；未配置时返回空掩码列表。

    需要环境变量提供权重：
    - LANGSAM_SAM_CKPT: sam 权重路径
    - LANGSAM_GDINO_CKPT: grounding-dino base 权重路径
    """
    sam_ckpt = os.environ.get("LANGSAM_SAM_CKPT")
    gdino_ckpt = os.environ.get("LANGSAM_GDINO_CKPT")
    if not (sam_ckpt and gdino_ckpt):
        return []
    try:
        from lang_sam import LangSAM  # type: ignore
    except Exception:
        return []

    try:
        model = LangSAM(
            sam_ckpt_path=sam_ckpt,
            gdino_model_ckpt_path=gdino_ckpt,
            gdino_processor_ckpt_path=gdino_ckpt,
        )
        masks = []
        for frame in frames:
            # 接口占位：真实实现依赖具体输入类型
            masks.append(0)
        return masks
    except Exception:
        return []
