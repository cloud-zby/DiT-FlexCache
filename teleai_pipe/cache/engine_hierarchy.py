from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import os
import pickle
from .log import cache_logger as logger
import math
from .config import FlexCacheConfig
from .decision import determine_skip_step
from .store import InMemoryKVStore, KVStore
from .encoder import encode_text, VECTOR_DIM, decompose_prompt
from .adapters.qdrant_adapter import QdrantVectorStore

class HierarchyCacheEngine:

    def __init__(
        self,
        qdrant_url,
        qdrant_api_key,
        qdrant_prefer_grpc,
        qdrant_request_timeout,
        text_vector_dim:int = 64,
    ):
        # 向量检索后端
        self._vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=qdrant_prefer_grpc,
            timeout=qdrant_request_timeout,
        )
        # 文本编码后的向量维度
        self._text_vector_dim = text_vector_dim

    @staticmethod
    def _project_to_dim(vec: List[float], dim: int) -> List[float]:
        if not isinstance(vec, list):
            vec = list(vec)
        if len(vec) == dim:
            return vec
        if dim <= 0:
            return vec
        buckets = [[] for _ in range(dim)]
        for idx, v in enumerate(vec):
            buckets[idx % dim].append(float(v))
        proj = [sum(b) / len(b) if b else 0.0 for b in buckets]
        norm = math.sqrt(sum(x * x for x in proj)) or 1.0
        return [x / norm for x in proj]

    def lookup(self, prompt: str) -> Dict[str, Any]:
        text_vec = self._project_to_dim(encode_text(prompt), self._text_vector_dim)
        # TODO
        text_res = self._vector_store.search(collection_name, text_vec, limit=1)
        text_top = text_res[0]
        
        return {
            "hit": True,
            "skip_step": skip_step,
            "cache_type": "exact",
            "similarity": 1.0,
            "latent_state": latent_state,
        }
