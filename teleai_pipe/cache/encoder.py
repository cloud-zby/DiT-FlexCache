from __future__ import annotations

from typing import Dict, Iterable, List
import os
import hashlib
import math

VECTOR_DIM = 64

_CLIP = None 
_LLM = None 


def decompose_prompt(prompt: str) -> Dict[str, str]:
    """提示词分解：
    - 首选 LLM（当环境提供 LLM_MODEL_PATH 且 transformers 可用时）；
    - 回退到启发式：把包含地点/环境的短语划到 background，其余为 object。
    """
    try:
        model_path = os.environ.get("LLM_MODEL_PATH")
        if model_path:
            _ensure_llm(model_path)
            return _llm_decompose(prompt)
    except Exception:
        pass
    return _heuristic_decompose(prompt)


def encode_text(prompt: str) -> List[float]:
    """文本向量化：
    - 首选 CLIP（当环境提供 CLIP_MODEL_PATH 且 transformers 可用时），并投影到 64 维；
    - 回退到 64 维哈希嵌入。
    """
    try:
        model_path = os.environ.get("CLIP_MODEL_PATH")
        if model_path:
            _ensure_clip(model_path)
            clip_vec = _clip_embed(prompt)
            return _project_to_dim(clip_vec, VECTOR_DIM)
    except Exception:
        pass
    return _hash_embed(prompt, VECTOR_DIM)


# -------- helpers --------

def _hash_embed(text: str, dim: int) -> List[float]:
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    for i in range(4):
        h = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
        for j in range(dim):
            vec[j] += h[j % len(h)] / 255.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _project_to_dim(vec: List[float], dim: int) -> List[float]:
    if not vec:
        return [0.0] * dim
    if len(vec) == dim:
        return vec
    # 简单分桶平均，将高维投影到 dim 维
    import itertools

    buckets = [[] for _ in range(dim)]
    for idx, v in enumerate(vec):
        buckets[idx % dim].append(float(v))
    proj = [sum(b) / len(b) if b else 0.0 for b in buckets]
    # 归一化
    norm = math.sqrt(sum(x * x for x in proj)) or 1.0
    return [x / norm for x in proj]


def _ensure_clip(model_path: str) -> None:
    global _CLIP
    if _CLIP is not None:
        return
    from transformers import CLIPModel, CLIPTokenizer  # type: ignore

    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    _CLIP = (model, tok)


def _clip_embed(text: str) -> List[float]:
    import torch

    model, tok = _CLIP
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    vec = out[0].cpu().float().tolist()
    return vec


def _ensure_llm(model_path: str) -> None:
    global _LLM
    if _LLM is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    _LLM = (model, tok)


def _llm_decompose(prompt: str) -> Dict[str, str]:
    # 轻量模板：不追求生成质量，只要结构化输出
    model, tok = _LLM
    system = (
        "Extract two fields from the prompt as JSON with keys 'object' and 'background'."
    )
    text = f"{system}\nPrompt: {prompt}\nOutput: {{'object': '...', 'background': '...'}}"
    import torch

    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    resp = tok.decode(out[0], skip_special_tokens=True)
    # 失败则回退heuristic
    obj, bg = None, None
    try:
        sub = resp.split("Output:")[-1]
        sub = sub[sub.find("{") : sub.find("}") + 1]
        d = eval(sub, {}, {})  # 仅解析简单字面量
        obj = str(d.get("object", "")).strip()
        bg = str(d.get("background", "")).strip()
    except Exception:
        pass
    if not obj and not bg:
        return _heuristic_decompose(prompt)
    return {"object": obj or prompt, "background": bg or ""}


def _heuristic_decompose(prompt: str) -> Dict[str, str]:
    p = prompt or ""
    bg_markers = [" in ", " at ", " on ", ",", ";"]
    for m in bg_markers:
        if m in p:
            left, right = p.split(m, 1)
            return {"object": left.strip() or p, "background": right.strip()}
    return {"object": p, "background": ""}
