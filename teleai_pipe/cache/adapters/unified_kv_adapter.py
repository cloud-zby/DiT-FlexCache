from __future__ import annotations

from ..store import KVStore
from ..log import cache_logger as logger


class UnifiedKVStore(KVStore): 
    def __init__(self, config_path: str):
        try:
            from kvcache_api_layer.backends.unified import UnifiedKVCacheStore
            from kvcache_api_layer.config import FluxonKvClientConfig
        except Exception as e:
            raise ImportError("kvcache_api_layer is not available") from e

        cfg = FluxonKvClientConfig.from_file(config_path)
        result = UnifiedKVCacheStore.new(cfg)
        if result.error() is not None:
            raise RuntimeError(f"UnifiedKV init failed: {result.error()}")
        self._store = result.success()
        logger.info("cache_reuse.kv.unified.init_ok")

    def get(self, key: str):
        r = self._store.get(key).unwrap().wait()
        if r.error() is not None:
            logger.debug(f"cache_reuse.kv.get miss key={key}")
            return None
        blob = r.success().bytes().success()
        return bytes(blob)

    def put(self, key: str, value: bytes) -> None:
        r = self._store.put(key, value).unwrap().wait()
        if r.error() is not None:
            raise RuntimeError(r.error())
        logger.debug(f"cache_reuse.kv.put key={key} size={len(value)}")

    def remove(self, key: str) -> None:
        r = self._store.remove(key).unwrap().wait()
        if r.error() is not None:
            raise RuntimeError(r.error())
        logger.info(f"cache_reuse.kv.remove key={key}")

    def get_size(self, key: str) -> int:
        r = self._store.get_size(key).unwrap().wait()
        if r.error() is not None:
            return 0
        return int(r.value())

    def is_exist(self, key: str) -> bool:
        r = self._store.is_exist(key).unwrap().wait()
        if r.error() is not None:
            return False
        return bool(r.value())
