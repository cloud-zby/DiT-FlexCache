from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class FlexCacheConfig:
    cache_dir: str = "./flexcache"
    collection_name: str = "wan_video_cache"
    collection_name_whole: str | None = None
    collection_name_background: str | None = None
    collection_name_object: str | None = None
    max_cache_size_gb: int = 1024

    device: str = "cuda"

    qdrant_url: Optional[str] = os.getenv("QDRANT_URL") or "http://127.0.0.1:6333"
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    qdrant_prefer_grpc: bool = False
    qdrant_request_timeout: int = 30

    kv_cache_config_path: Optional[str] = os.getenv("FLEXCACHE_KV_CONFIG") or None

    similarity_threshold: float = 0.65
    decomposed_similarity_threshold: float = 1.0
    video_similarity_threshold: float = 0.65

    key_steps: List[int] = field(default_factory=lambda: [0, 1, 5, 10, 15, 20, 25])

    @classmethod
    def from_kwargs(cls, **kwargs) -> "FlexCacheConfig":
        cfg = cls()
        for k, v in kwargs.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        base = cfg.collection_name
        if not cfg.collection_name_whole:
            cfg.collection_name_whole = f"{base}_whole"
        if not cfg.collection_name_background:
            cfg.collection_name_background = f"{base}_background"
        if not cfg.collection_name_object:
            cfg.collection_name_object = f"{base}_object"
        return cfg
