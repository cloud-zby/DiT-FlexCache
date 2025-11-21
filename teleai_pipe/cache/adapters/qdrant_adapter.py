from __future__ import annotations
from typing import Iterable, List, Sequence
from loguru import logger

from ..store import VectorPoint, VectorStore
from ..log import cache_logger as logger


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, api_key: str | None = None, prefer_grpc: bool = False, timeout: int = 30):
        try:
            from qdrant_client import QdrantClient  
        except Exception as e:  
            raise ImportError("qdrant_client is not available") from e
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=timeout)

    def ensure_collection(self, collection: str, size: int) -> None:  
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff  

        try:
            cols = self._client.get_collections().collections
            if any(c.name == collection for c in cols):
                logger.debug(f"cache_reuse.vector.ensure_collection exists name={collection} size={size}")
                return
        except Exception:
            pass
        try:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10000),
                ),
            )
            logger.info(f"cache_reuse.vector.ensure_collection created name={collection} size={size}")
        except Exception:
            pass

    def upsert(self, collection: str, points: List[VectorPoint]) -> None:  
        from qdrant_client.models import PointStruct 

        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=p.id, vector=list(p.vector), payload=p.payload) for p in points],
        )
        logger.debug(f"cache_reuse.vector.upsert collection={collection} count={len(points)}")

    def search(self, collection: str, query_vector: Sequence[float], limit: int = 1):  
        res = self._client.search(collection_name=collection, query_vector=list(query_vector), limit=limit)
        top = res[0] if res else None
        score = float(getattr(top, "score", 0.0)) if top is not None else 0.0
        logger.debug(f"cache_reuse.vector.search collection={collection} limit={limit} top_score={score:.3f}")
        return res

    def delete(self, collection: str, ids: Iterable[str]) -> None:  
        from qdrant_client import models 

        self._client.delete(collection_name=collection, points_selector=models.PointIdsList(points=list(ids)))
        logger.info(f"cache_reuse.vector.delete collection={collection} count={len(list(ids))}")

    def get_vector_size(self, collection: str) -> int | None:  
        try:
            info = self._client.get_collection(collection)
        except Exception:
            return None

        size = None
        try:
            cfg = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
            size = getattr(cfg, "size", None)
            if size is None:
                configs = getattr(cfg, "configs", None)
                if isinstance(configs, dict):
                    for _name, vp in configs.items():
                        size = getattr(vp, "size", None)
                        if size is not None:
                            break
            if size is None and isinstance(cfg, dict):
                for _name, vp in cfg.items():
                    size = getattr(vp, "size", None)
                    if size is not None:
                        break
        except Exception:
            size = None

        try:
            logger.debug(
                f"cache_reuse.vector.collection_dim name={collection} dim={int(size) if size is not None else None}"
            )
        except Exception:
            pass
        try:
            return int(size) if size is not None else None
        except Exception:
            return None
