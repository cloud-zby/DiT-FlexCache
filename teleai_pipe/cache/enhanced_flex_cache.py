from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass

from .log import cache_logger as logger

from .config import FlexCacheConfig
from .decision import determine_skip_step
from .engine_lite import LiteFlexCacheEngine
from .engine_hierarchy import HierarchyCacheEngine


@dataclass
class _CacheResult:
    hit: bool
    skip_step: int = 0
    cache_type: str = "none"
    similarity: float = 0.0
    latent_state: Any = None
    cached_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit": self.hit,
            "skip_step": self.skip_step,
            "cache_type": self.cache_type,
            "similarity": self.similarity,
            "latent_state": self.latent_state,
            "cached_prompt": self.cached_prompt,
        }


class _DisabledEngine:
    """在依赖缺失或未配置时的降级引擎。"""

    def __init__(self, key_steps: Iterable[int]):
        self.key_steps = list(key_steps)

    def lookup(self, prompt: str) -> Dict[str, Any]:
        logger.debug("FlexCache disabled: lookup returns miss")
        return _CacheResult(hit=False).to_dict()

    def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, Any],
        timesteps,
        num_frames: int,
        generated_video_frames: Optional[List[Any]] = None,
        generated_video_tensor: Optional[Any] = None,
    ) -> None:
        logger.debug("FlexCache disabled: save is no-op")


class FlexCache:
    """统一对外 FlexCache 实现。
    """

    def __init__(
        self,
        cache_dir: str = "./flexcache",
        collection_name: str = "wan_video_cache",
        device: str = "cuda",
        max_cache_size_gb: int = 10,
        kv_cache_config_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        request_timeout: int = 30,
        key_steps: Optional[Iterable[int]] = None,
        prefer_legacy: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        cfg = FlexCacheConfig.from_kwargs(
            cache_dir=cache_dir,
            collection_name=collection_name,
            device=device,
            max_cache_size_gb=max_cache_size_gb,
            kv_cache_config_path=kv_cache_config_path,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_prefer_grpc=prefer_grpc,
            qdrant_request_timeout=request_timeout,
        )
        if similarity_threshold is not None:
            try:
                cfg.similarity_threshold = float(similarity_threshold)
            except Exception:
                pass
        if key_steps is not None:
            cfg.key_steps = list(key_steps)

        self.config = cfg
        self.key_steps = list(cfg.key_steps)

        if prefer_legacy is not None:
            logger.info("FlexCache: 'prefer_legacy' is deprecated and ignored; using package Lite engine.")

        try:
            engine = HierarchyCacheEngine(cfg)
            # engine = LiteFlexCacheEngine(cfg)
        except Exception as e:  # pragma: no cover
            logger.info(f"Lite engine unavailable, fallback to disabled: {e}")
            engine = _DisabledEngine(key_steps=self.key_steps)

        self._engine = engine

    def lookup(self, prompt: str) -> Dict[str, Any]:
        return self._engine.lookup(prompt)

    def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, Any],
        timesteps,
        num_frames: int,
        generated_video_frames: Optional[List[Any]] = None,
        generated_video_tensor: Optional[Any] = None,
    ) -> None:
        return self._engine.save(
            prompt,
            latent_states_dict,
            timesteps,
            num_frames,
            generated_video_frames,
            generated_video_tensor,
        )
