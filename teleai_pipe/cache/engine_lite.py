from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import os
import pickle
from .log import cache_logger as logger

from .config import FlexCacheConfig
from .decision import determine_skip_step
from .store import InMemoryKVStore, KVStore
from .encoder import encode_text, VECTOR_DIM, decompose_prompt


@dataclass
class _IndexEntry:
    cache_id: str
    prompt: str
    saved_steps: List[int]


class LiteFlexCacheEngine:
    def __init__(self, cfg: FlexCacheConfig, kv: Optional[KVStore] = None) -> None:
        self.cfg = cfg
        self.key_steps = list(cfg.key_steps)
        # 标记向量检索是否已失效（首次失败后禁用）
        self._vstore_failed: bool = False
        # 优先使用 UnifiedKV（若配置与依赖可用），否则使用内存 KV
        if kv is not None:
            self.kv = kv
        else:
            self.kv = self._init_kv_from_cfg(cfg)
        self.index_path = os.path.join(cfg.cache_dir, "index.pkl")
        self.meta_path = os.path.join(cfg.cache_dir, "cache_meta.pkl")
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self._index: Dict[str, _IndexEntry] = self._load_index()
        self._meta: Dict[str, Dict[str, Any]] = self._load_meta()
        # 可选向量检索后端
        self._vstore = self._init_vector_store_from_cfg(cfg)
        if self._vstore is not None:
            try:
                self._vstore.ensure_collection(self.cfg.collection_name_whole, VECTOR_DIM)  # type: ignore[attr-defined]
                self._vstore.ensure_collection(self.cfg.collection_name_background, VECTOR_DIM)  # type: ignore[attr-defined]
                self._vstore.ensure_collection(self.cfg.collection_name_object, VECTOR_DIM)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._vdim_whole = self._get_collection_dim(self.cfg.collection_name_whole) or VECTOR_DIM
            self._vdim_bg = self._get_collection_dim(self.cfg.collection_name_background) or VECTOR_DIM
            self._vdim_obj = self._get_collection_dim(self.cfg.collection_name_object) or VECTOR_DIM
        else:
            self._vdim_whole = VECTOR_DIM
            self._vdim_bg = VECTOR_DIM
            self._vdim_obj = VECTOR_DIM
        try:
            kv_name = type(self.kv).__name__
            vstore_name = type(self._vstore).__name__ if self._vstore is not None else "None"
            logger.info(
                f"cache_reuse.engine=lite kv={kv_name} vstore={vstore_name} collections=({self.cfg.collection_name_whole}:{self._vdim_whole},{self.cfg.collection_name_background}:{self._vdim_bg},{self.cfg.collection_name_object}:{self._vdim_obj})"
            )
            logger.info(
                f"cache_reuse.config key_steps={self.key_steps} similarity_threshold={cfg.similarity_threshold}"
            )
        except Exception:
            pass

    # ------------ index -------------
    def _load_index(self) -> Dict[str, _IndexEntry]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.debug(f"cache_reuse.index_load_error: {e}")
        return {}

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.debug(f"cache_reuse.index_save_error: {e}")

    # ------------ metadata -------------
    def _load_meta(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "rb") as f:
                    data = pickle.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.debug(f"cache_reuse.meta_load_error: {e}")
        return {}

    def _save_meta(self) -> None:
        try:
            with open(self.meta_path, "wb") as f:
                pickle.dump(self._meta, f)
        except Exception as e:
            logger.debug(f"cache_reuse.meta_save_error: {e}")

    def _update_cache_access(self, cache_id: str) -> None:
        meta = self._meta.get(cache_id)
        if meta is None:
            return
        import time as _t

        meta["access_count"] = int(meta.get("access_count", 0)) + 1
        meta["last_access_time"] = float(_t.time())
        self._save_meta()

    def _ensure_capacity(self, required_mb: float) -> None:
        limit_mb = float(self.cfg.max_cache_size_gb) * 1024.0
        now_total = sum(float(v.get("size_mb", 0.0)) for v in self._meta.values())
        if now_total + required_mb <= limit_mb:
            return
        need = now_total + required_mb - limit_mb
        logger.info(
            f"cache_reuse.evict start need={need:.2f}MB total={now_total:.2f}MB limit={limit_mb:.2f}MB"
        )
        # LRU（last_access_time 最小优先驱逐）
        items = sorted(
            self._meta.items(), key=lambda kv: float(kv[1].get("last_access_time", 0.0))
        )
        freed = 0.0
        for cache_id, meta in items:
            freed += float(meta.get("size_mb", 0.0))
            self._evict_cache(cache_id, meta)
            if freed >= need:
                break
        total = sum(float(v.get("size_mb", 0.0)) for v in self._meta.values())
        logger.info(f"cache_reuse.evict done freed={freed:.2f}MB total_now={total:.2f}MB")

    def _evict_cache(self, cache_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"cache_reuse.evict item cache_id={cache_id}")
        # 删除 KV: steps + masks + video
        steps = list(meta.get("steps", [])) if isinstance(meta, dict) else []
        for s in steps:
            try:
                self.kv.remove(f"{cache_id}_step{int(s)}")
            except Exception:
                pass
        # 预留：掩码与视频
        for k in [f"{cache_id}_object_mask", f"{cache_id}_background_mask", f"{cache_id}_video_tensor"]:
            try:
                self.kv.remove(k)
            except Exception:
                pass
        # 删除向量集合的点
        if getattr(self, "_vstore", None) is not None and not self._vstore_failed:
            try:
                self._vstore.delete(self.cfg.collection_name_whole, [cache_id])
                self._vstore.delete(self.cfg.collection_name_background, [cache_id])
                self._vstore.delete(self.cfg.collection_name_object, [cache_id])
            except Exception:
                pass
        # 删除索引中的 prompt→entry（若匹配此 cache_id）
        to_del = [p for p, e in self._index.items() if getattr(e, "cache_id", None) == cache_id]
        for p in to_del:
            self._index.pop(p, None)
        self._save_index()
        # 删除元数据
        self._meta.pop(cache_id, None)
        self._save_meta()

    def _init_kv_from_cfg(self, cfg: FlexCacheConfig) -> KVStore:
        if cfg.kv_cache_config_path:
            try:
                from .adapters.unified_kv_adapter import UnifiedKVStore

                return UnifiedKVStore(cfg.kv_cache_config_path)
            except Exception as e:
                logger.info(
                    f"cache_reuse.kv.unified_unavailable: fallback=InMemory reason={e}"
                )
        return InMemoryKVStore()

    # ------------ api -------------
    def _get_collection_dim(self, name: str) -> int | None:
        try:
            return int(self._vstore.get_vector_size(name))  # type: ignore[attr-defined]
        except Exception:
            return None

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
        # 归一化
        import math as _m

        norm = _m.sqrt(sum(x * x for x in proj)) or 1.0
        return [x / norm for x in proj]
    def lookup(self, prompt: str) -> Dict[str, Any]:
        # 若有向量检索后端且未处于失效状态，优先进行近似搜索（三集合）
        if getattr(self, "_vstore", None) is not None and not self._vstore_failed:
            try:
                whole_vec = self._project_to_dim(encode_text(prompt), self._vdim_whole)
                parts = decompose_prompt(prompt)
                bg_prompt = parts.get("background", "")
                obj_prompt = parts.get("object", prompt)
                bg_vec = self._project_to_dim(encode_text(bg_prompt), self._vdim_bg)
                obj_vec = self._project_to_dim(encode_text(obj_prompt), self._vdim_obj)

                # 三集合检索
                whole_res = self._vstore.search(self.cfg.collection_name_whole, whole_vec, limit=1)
                bg_res = self._vstore.search(self.cfg.collection_name_background, bg_vec, limit=1)
                obj_res = self._vstore.search(self.cfg.collection_name_object, obj_vec, limit=1)

                def _top(r):
                    return r[0] if r else None

                whole_top = _top(whole_res)
                bg_top = _top(bg_res)
                obj_top = _top(obj_res)

                def _score(x, default=0.0):
                    if x is None:
                        return 0.0
                    try:
                        return float(getattr(x, "score", 0.0) or (x.get("score") if isinstance(x, dict) else 0.0))
                    except Exception:
                        return default

                s_whole = _score(whole_top)
                s_bg = _score(bg_top)
                s_obj = _score(obj_top)

                try:
                    logger.info(
                        f"cache_reuse.vector.scores whole={s_whole:.3f} bg={s_bg:.3f} obj={s_obj:.3f} thresholds={{whole:{self.cfg.similarity_threshold},decomp:{self.cfg.decomposed_similarity_threshold}}}"
                    )
                except Exception:
                    pass

                def _payload(x):
                    if x is None:
                        return {}
                    return getattr(x, "payload", None) or (x.get("payload") if isinstance(x, dict) else {})

                # Whole 命中
                min_decomp = min(s_bg, s_obj) if (s_bg >= self.cfg.decomposed_similarity_threshold and s_obj >= self.cfg.decomposed_similarity_threshold) else 0.0
                if s_whole >= self.cfg.similarity_threshold and s_whole >= min_decomp and whole_top is not None:
                    payload = _payload(whole_top)
                    cache_id = getattr(whole_top, "id", None) or (whole_top.get("id") if isinstance(whole_top, dict) else None)
                    saved_steps = payload.get("saved_steps", [])
                    skip_step = determine_skip_step(s_whole, saved_steps)
                    if cache_id and skip_step > 0:
                        for s in [int(skip_step)] + sorted([int(x) for x in saved_steps if int(x) < int(skip_step)], reverse=True):
                            latent_key = f"{cache_id}_step{s}"
                            blob = self.kv.get(latent_key)
                            if blob is not None:
                                try:
                                    logger.info(f"cache_reuse.kv.get.ok key={latent_key} bytes={len(blob)}")
                                except Exception:
                                    pass
                                latent_state = pickle.loads(blob)
                                try:
                                    logger.info(f"cache_reuse.hit vector=whole skip={s} score={s_whole:.3f}")
                                except Exception:
                                    pass
                                # 访问打点
                                self._update_cache_access(str(cache_id))
                                return {
                                    "hit": True,
                                    "skip_step": int(s),
                                    "cache_type": "vector_whole",
                                    "similarity": s_whole,
                                    "latent_state": latent_state,
                                    "cached_prompt": payload.get("prompt", ""),
                                }

                # Decomposed 命中
                if (s_bg >= self.cfg.decomposed_similarity_threshold and s_obj >= self.cfg.decomposed_similarity_threshold and bg_top is not None and obj_top is not None):
                    payload_bg = _payload(bg_top)
                    payload_obj = _payload(obj_top)
                    cache_id_bg = getattr(bg_top, "id", None) or (bg_top.get("id") if isinstance(bg_top, dict) else None)
                    cache_id_obj = getattr(obj_top, "id", None) or (obj_top.get("id") if isinstance(obj_top, dict) else None)
                    steps_bg = set(int(x) for x in payload_bg.get("saved_steps", []))
                    steps_obj = set(int(x) for x in payload_obj.get("saved_steps", []))
                    common = sorted(list((steps_bg & steps_obj & set(self.key_steps))), reverse=True)
                    if common:
                        decision = determine_skip_step(min(s_bg, s_obj), common)
                        candidates = [int(decision)] + [s for s in common if s < int(decision)]
                        for s in candidates:
                            for cid in [cache_id_bg, cache_id_obj]:
                                if cid:
                                    latent_key = f"{cid}_step{s}"
                                    blob = self.kv.get(latent_key)
                                    if blob is not None:
                                        try:
                                            logger.info(f"cache_reuse.kv.get.ok key={latent_key} bytes={len(blob)}")
                                        except Exception:
                                            pass
                                        latent_state = pickle.loads(blob)
                                        try:
                                            logger.info(
                                                f"cache_reuse.hit vector=decomposed skip={s} score=min({s_bg:.3f},{s_obj:.3f})"
                                            )
                                        except Exception:
                                            pass
                                        # 访问打点
                                        self._update_cache_access(str(cid))
                                        return {
                                            "hit": True,
                                            "skip_step": int(s),
                                            "cache_type": "vector_decomposed",
                                            "similarity": float(min(s_bg, s_obj)),
                                            "latent_state": latent_state,
                                            "cached_prompt": f"Background: {payload_bg.get('prompt','')} + Object: {payload_obj.get('prompt','')}",
                                        }
            except Exception as e:
                # 首次失败时记录告警，并在本进程内禁用向量检索（直到重启）
                if not self._vstore_failed:
                    self._vstore_failed = True
                    try:
                        logger.warning(
                            f"cache_reuse.vector_store.unavailable: {e!s}; disable until restart"
                        )
                    except Exception:
                        pass

        # 精确匹配策略（无 embedding 与检索）
        entry = self._index.get(prompt)
        if not entry:
            return {"hit": False}

        # 选择最大可跳过步（严格等价与 determine_skip_step 规则）
        skip_step = determine_skip_step(1.0, entry.saved_steps)
        if skip_step <= 0:
            return {"hit": False}

        # 取出 latent
        latent_key = f"{entry.cache_id}_step{skip_step}"
        try:
            blob = self.kv.get(latent_key)
            if blob is None:
                return {"hit": False}
            try:
                logger.info(f"cache_reuse.kv.get.ok key={latent_key} bytes={len(blob)}")
            except Exception:
                pass
            latent_state = pickle.loads(blob)
        except Exception as e:
            logger.debug(f"cache_reuse.latent_load_error: {e}")
            return {"hit": False}

        # 访问打点（精确匹配）
        self._update_cache_access(str(entry.cache_id))
        return {
            "hit": True,
            "skip_step": skip_step,
            "cache_type": "exact",
            "similarity": 1.0,
            "latent_state": latent_state,
            "cached_prompt": prompt,
        }
        # unreachable

    def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, Any],
        timesteps,
        num_frames: int,
        generated_video_frames: Optional[List[Any]] = None,
        generated_video_tensor: Optional[Any] = None,
    ) -> None:
        # 分配 cache_id（哈希）
        import hashlib
        import uuid

        cache_id = str(uuid.uuid4())
        qdrant_point_id = cache_id
        saved_steps: List[int] = []

        # 计算即将写入的占用大小，用于容量控制
        blobs: List[tuple[int, bytes]] = []
        for step, latent in latent_states_dict.items():
            if int(step) not in self.key_steps:
                continue
            try:
                b = pickle.dumps(latent)
            except Exception:
                continue
            blobs.append((int(step), b))
        total_bytes = sum(len(b) for _, b in blobs)
        total_mb = total_bytes / (1024 * 1024)
        try:
            logger.debug(
                f"cache_reuse.save.prepare cache_id={cache_id} steps={[s for s,_ in blobs]} size_mb={total_mb:.2f}"
            )
        except Exception:
            pass
        self._ensure_capacity(total_mb)

        for step, blob in blobs:
            try:
                latent_key = f"{cache_id}_step{int(step)}"
                self.kv.put(latent_key, blob)
                saved_steps.append(int(step))
                try:
                    logger.debug(f"cache_reuse.kv.put ok key={latent_key}")
                except Exception:
                    pass
            except Exception as e:
                msg = str(e)
                if "No space left" in msg:
                    try:
                        logger.warning(
                            f"cache_reuse.kv_store.put_failed_no_space step={step}: {e}"
                        )
                    except Exception:
                        pass
                else:
                    logger.debug(f"cache_reuse.latent_save_error step={step} reason={e}")

        if saved_steps:
            self._index[prompt] = _IndexEntry(
                cache_id=cache_id, prompt=prompt, saved_steps=sorted(set(saved_steps))
            )
            self._save_index()
            # 写入向量检索后端
            if getattr(self, "_vstore", None) is not None and not self._vstore_failed:
                try:
                    from .store import VectorPoint
                    parts = decompose_prompt(prompt)
                    bg_prompt = parts.get("background", "")
                    obj_prompt = parts.get("object", prompt)
                    vec_whole = self._project_to_dim(encode_text(prompt), self._vdim_whole)
                    vec_bg = self._project_to_dim(encode_text(bg_prompt), self._vdim_bg)
                    vec_obj = self._project_to_dim(encode_text(obj_prompt), self._vdim_obj)

                    whole_point = VectorPoint(
                        id=qdrant_point_id,
                        vector=vec_whole,
                        payload={
                            "prompt": prompt,
                            "saved_steps": sorted(set(saved_steps)),
                            "num_frames": int(num_frames),
                            "original_prompt": prompt,
                        },
                    )
                    bg_point = VectorPoint(
                        id=qdrant_point_id,
                        vector=vec_bg,
                        payload={
                            "prompt": bg_prompt,
                            "saved_steps": sorted(set(saved_steps)),
                            "num_frames": int(num_frames),
                            "original_prompt": prompt,
                        },
                    )
                    obj_point = VectorPoint(
                        id=qdrant_point_id,
                        vector=vec_obj,
                        payload={
                            "prompt": obj_prompt,
                            "saved_steps": sorted(set(saved_steps)),
                            "num_frames": int(num_frames),
                            "original_prompt": prompt,
                        },
                    )
                    self._vstore.upsert(self.cfg.collection_name_whole, [whole_point])
                    self._vstore.upsert(self.cfg.collection_name_background, [bg_point])
                    self._vstore.upsert(self.cfg.collection_name_object, [obj_point])
                    try:
                        logger.info(
                            f"cache_reuse.vector.upsert ok steps={sorted(set(saved_steps))}"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"cache_reuse.vector_upsert_failed: {e}")

            # 更新元数据并持久化
            import time as _t
            self._meta[cache_id] = {
                "prompt": prompt,
                "steps": sorted(set(saved_steps)),
                "num_frames": int(num_frames),
                "size_mb": float(total_mb),
                "access_count": 0,
                "last_access_time": float(_t.time()),
            }
            self._save_meta()

    def _init_vector_store_from_cfg(self, cfg: FlexCacheConfig):
        if cfg.qdrant_url:
            try:
                from .adapters.qdrant_adapter import QdrantVectorStore

                return QdrantVectorStore(
                    url=cfg.qdrant_url,
                    api_key=cfg.qdrant_api_key,
                    prefer_grpc=cfg.qdrant_prefer_grpc,
                    timeout=cfg.qdrant_request_timeout,
                )
            except Exception as e:
                logger.info(
                    f"cache_reuse.vector_store.unavailable: disable reason={e}"
                )
        return None
