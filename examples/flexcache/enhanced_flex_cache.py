import torch
import numpy as np
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)
import uuid
import os
import time
import pickle
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import cv2
from typing import Dict, List, Tuple, Optional, Union
import json
import time
import logging
import importlib
import hashlib
import tempfile
from loguru import logger 
try:
    from kvcache_api_layer.backends.unified import UnifiedKVCacheStore, UnifiedMemHolder
    from kvcache_api_layer.config import FluxonKvClientConfig
    from kvcache_api_layer.api_error import Result, ApiError, GeneralError

    UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    logger.info(
        "Warning: UnifiedKVCacheStore modules not available, falling back to local storage"
    )
    UNIFIED_CACHE_AVAILABLE = False


model_root = "/nvfile-heatstorage/ai_infra/ckpts/liw57/zgf/file"


class FlexCache:
    def __init__(
        self,
        cache_dir="./flexcache",
        collection_name="wan_video_cache",
        device="cuda",
        max_cache_size_gb=1024,
        kv_cache_config_path='/gemini/space/jinyx5/repo/teleai_pipe/examples/flexcache/KVstore.yaml',
        qdrant_url: str = "http://127.0.0.1:6333",
        qdrant_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        request_timeout: int = 30,
    ):
        self.cache_dir = cache_dir
        self.device = device
        self.max_cache_size_gb = max_cache_size_gb

        # self.client = QdrantClient(path=os.path.join(cache_dir, "qdrant"))
        self.client = QdrantClient(
            # url=qdrant_url,
            host="localhost", port=6333,
            api_key=qdrant_api_key,
            prefer_grpc=prefer_grpc,
            timeout=request_timeout,
        )

        try:
            _ = self.client.get_collections()
        except Exception as e:
            raise RuntimeError(
                f"Qdrant not reachable at {qdrant_url}. "
                f"Please ensure the server is running and ports are open. Details: {e}"
            )

        # 三个集合
        self.collection_name_whole = collection_name + "_whole"
        self.collection_name_background = collection_name + "_background"
        self.collection_name_object = collection_name + "_object"
        self.collection_name_video = collection_name + "_video"

        # CLIP
        self.clip_model = CLIPModel.from_pretrained(
            "/gemini/platform/public/weights/clip-vit-base-patch32/"
        ).to(device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            "/gemini/platform/public/weights/clip-vit-base-patch32/"
        )
        self.clip_model.eval()
        logger.info("clip_model completed")

        # Llama2-7B
        local_llama_path = "/gemini/platform/public/weights/Llama-2-7b-hf"

        # Bind model to a single device. Prefer LOCAL_RANK when present.
        if device == "cuda" and torch.cuda.is_available():
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            except Exception:
                local_rank = torch.cuda.current_device()
            target_device = f"cuda:{local_rank}"
        else:
            target_device = device
        self.llama_model = LlamaForCausalLM.from_pretrained(
            local_llama_path,
            torch_dtype=torch.float16,
            # device_map="auto",
            local_files_only=True,
        ).to(target_device)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            local_llama_path, local_files_only=True
        )
        self.llama_model.eval()
        logger.info("llama_model completed")

        self.key_steps = [0,1]

        self.similarity_threshold = 0.65
        self.decomposed_similarity_threshold = 1.00
        self.video_similarity_threshold = 0.65

        self.cache_metadata = self._load_cache_metadata()

        self._init_collections()

        # LangSAM
        self._init_langsam()

        self.kv_cache_store = None
        if kv_cache_config_path:
            try:
                
                from kvcache_api_layer.backends.unified import UnifiedKVCacheStore, UnifiedMemHolder
                from kvcache_api_layer.config import FluxonKvClientConfig
                from kvcache_api_layer.api_error import Result, ApiError, GeneralError
                config = FluxonKvClientConfig.from_file(kv_cache_config_path)
                kv_cache_result = UnifiedKVCacheStore.new(config)
                if kv_cache_result.error() is not None:
                    logger.info(f"init KVCacheStore failed: {kv_cache_result.error()}")
                else:
                    self.kv_cache_store = kv_cache_result.success()
                    time.sleep(10)
                    logger.info("KVCacheStore init success")
            except Exception as e:
                logger.info(f"init KVCacheStore error: {e}")
        else:
            logger.info("can't find FluxonKvClientConfig file, error")
            self.kv_cache_store = None

    def _init_collections(self):
        collections = self.client.get_collections().collections
        exist = {c.name for c in collections}
        for name in [
            self.collection_name_whole,
            self.collection_name_background,
            self.collection_name_object,
            self.collection_name_video,
        ]:
            if name not in exist:
                try:
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=512,
                            distance=Distance.COSINE,
                            hnsw_config=models.HnswConfigDiff(
                                m=16,
                                ef_construct=256,
                                full_scan_threshold=10000,
                                max_indexing_threads=0,
                            ),
                        ),
                    )
                    logger.info(f"Created collection: {name}")
                except Exception as e:
                    msg = str(e).lower()
                    if "already exists" in msg or "exists" in msg:
                        logger.info(f"Collection already exists (race condition): {name}")
                    else:
                        raise

    def _init_langsam(self):
        """LangSAM"""
        try:
            from lang_sam import LangSAM

            gdino_model_root = "/workspace/model"
            self.langsam = LangSAM(
                sam_ckpt_path=f"{model_root}/sam2.1_hiera_small.pt",
                gdino_model_ckpt_path=f"{gdino_model_root}/grounding-dino-base",
                gdino_processor_ckpt_path=f"{gdino_model_root}/grounding-dino-base",
            )
            logger.info("LangSAM initialized successfully")
        except ImportError:
            logger.info("Warning: LangSAM not available. Please install lang-sam package.")
            self.langsam = None

    def decompose_prompt_with_llama(self, prompt: str) -> Dict[str, str]:
        """Llama2-7B"""
        # 改进的system prompt，更准确地区分object和background
        #         system_prompt = """Analyze this video prompt and extract the main subject/object and the environment/background. Return JSON with exactly these keys:
        # - "object": The main character, person, animal, or thing performing actions (include colors, actions)
        # - "background": The location, setting, environment, weather, time conditions

        # Examples:
        # Input: "A red car driving through a rainy city street at night"
        # Output: {"object": "red car driving", "background": "rainy city street at night"}

        # Input: "A cat sitting on a windowsill watching birds outside"
        # Output: {"object": "cat sitting and watching birds", "background": "windowsill, outside view"}

        # Input: "Children playing soccer in a green park on a sunny day"
        # Output: {"object": "children playing soccer", "background": "green park, sunny day"}

        # Now analyze:"""
        system_prompt = """Analyze this video prompt and extract the main subject/object and the environment/background. Return JSON with exactly these keys:
- "object": The main character, person, animal, or thing performing actions (include colors, actions in base form)- must be a string
- "background": The location, setting, environment, weather, time conditions- must be a string

Examples:
Input: "A red car driving through a rainy city street at night"
Output: {"object": "red car drive", "background": "rainy city street at night"}

Input: "A cat sitting on a windowsill watching birds outside"
Output: {"object": "cat sit and watch birds", "background": "windowsill, outside view"}

Input: "Children playing soccer in a green park on a sunny day"
Output: {"object": "children play soccer", "background": "green park, sunny day"}

Now analyze:"""

        full_prompt = f"{system_prompt} {prompt}\nOutput:"

        inputs = self.llama_tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # 减少长度
            padding=False,
        )
        device = next(self.llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 100,
                "min_new_tokens": 10,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": self.llama_tokenizer.pad_token_id,
                "eos_token_id": self.llama_tokenizer.eos_token_id,
                "num_return_sequences": 1,
                "early_stopping": False,
            }

            if self.device == "cuda":
                generate_kwargs["use_cache"] = True

            try:
                outputs = self.llama_model.generate(**inputs, **generate_kwargs)
            except Exception as e:
                logger.info(f"llama decompose prompt error: {e}")
                return self._simple_decompose(prompt)

        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.llama_tokenizer.decode(new_tokens, skip_special_tokens=True)

        if not response.strip():
            full_response = self.llama_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            if "Output:" in full_response:
                response = full_response.split("Output:")[-1].strip()

        response = response.strip().replace("\n", " ").replace("\r", "")

        if response:
            try:
                json_start = response.find("{")
                if json_start != -1:
                    brace_count = 0
                    json_end = json_start
                    for i, char in enumerate(response[json_start:], json_start):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > json_start:
                        json_str = response[json_start:json_end]
                        json_str = " ".join(json_str.split())
                        decomposed = json.loads(json_str)

                        if "object" in decomposed and "background" in decomposed:
                            result = {
                                "background": decomposed.get("background", ""),
                                "object": decomposed.get("object", ""),
                            }
                            logger.info(f"Prompt分解结果:")
                            logger.info(f"  Background: {result['background']}")
                            logger.info(f"  Object: {result['object']}")
                            return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.info(f"Failed to parse Llama response: {e}")

        return self._simple_decompose(prompt)

    def _simple_decompose(self, prompt: str) -> Dict[str, str]:
        """简单的提示分解作为回退"""
        words = prompt.split()
        mid_point = len(words) // 2
        result = {
            "background": " ".join(words[:mid_point]),
            "object": " ".join(words[mid_point:]),
        }
        logger.info(f"简单分解结果 (回退方案):")
        logger.info(f"  Background: {result['background']}")
        logger.info(f"  Object: {result['object']}")
        return result

    @torch.no_grad()
    def encode_prompt(self, prompt):
        inputs = self.clip_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)

        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().squeeze()

    def segment_video_with_langsam(
        self, video_frames: List[np.ndarray], object_prompt: str
    ) -> List[np.ndarray]:
        """使用LangSAM对视频进行分割，生成对象掩码"""
        if self.langsam is None:
            logger.info("LangSAM not available, using dummy masks")
            return [
                np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                for frame in video_frames
            ]

        masks = []
        for frame in video_frames:
            try:
                from PIL import Image

                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                image_pil = Image.fromarray(frame).convert("RGB")

                results = self.langsam.predict([image_pil], [object_prompt])

                if (
                    results
                    and len(results) > 0
                    and "masks" in results[0]
                    and len(results[0]["masks"]) > 0
                ):
                    mask = results[0]["masks"][0].astype(np.uint8) * 255
                else:
                    logger.info("没检测到对象")
                    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                masks.append(mask)
            except Exception as e:
                logger.info(f"Error in LangSAM segmentation: {e}")
                mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                masks.append(mask)

        return masks


    def lookup(self, prompt: str):
        """查找相似的缓存prompt，支持对象/背景分离"""
        decomposed = self.decompose_prompt_with_llama(prompt)
        background_prompt = decomposed["background"]
        object_prompt = decomposed["object"]

        # 使用CLIP对三种prompt进行编码
        whole_embedding = self.encode_prompt(prompt)
        background_embedding = self.encode_prompt(background_prompt)
        object_embedding = self.encode_prompt(object_prompt)

        search_params = models.SearchParams(hnsw_ef=256, exact=False)

        # 在三个集合中搜索最相似的缓存prompt
        whole_result = self.client.search(
            collection_name=self.collection_name_whole,
            query_vector=whole_embedding.tolist(),
            limit=1,
            search_params=search_params,
        )

        background_result = self.client.search(
            collection_name=self.collection_name_background,
            query_vector=background_embedding.tolist(),
            limit=1,
            search_params=search_params,
        )

        object_result = self.client.search(
            collection_name=self.collection_name_object,
            query_vector=object_embedding.tolist(),
            limit=1,
            search_params=search_params,
        )

        sim_whole = (
            whole_result[0].score
            if whole_result and whole_result[0].score >= self.similarity_threshold
            else 0
        )
        sim_background = (
            background_result[0].score
            if background_result
            and background_result[0].score >= self.decomposed_similarity_threshold
            else 0
        )
        sim_object = (
            object_result[0].score
            if object_result
            and object_result[0].score >= self.decomposed_similarity_threshold
            else 0
        )

        # sim_whole = whole_result[0].score if whole_result else 0
        # sim_background = background_result[0].score if background_result else 0
        # sim_object = object_result[0].score if object_result else 0

        min_decomposed_sim = (
            min(sim_background, sim_object)
            if (sim_background > 0 and sim_object > 0)
            else 0
        )
        logger.info(
            f"Similarity scores - Whole: {sim_whole:.3f}, Background: {sim_background:.3f}, Object: {sim_object:.3f}"
        )

        # 命中逻辑
        if sim_whole >= self.similarity_threshold and sim_whole >= min_decomposed_sim:
            return self._handle_whole_cache_hit(whole_result[0], sim_whole)
        elif min_decomposed_sim >= self.decomposed_similarity_threshold:
            return self._handle_decomposed_cache_hit(
                background_result[0],
                object_result[0],
                sim_background,
                sim_object,
                background_prompt,
                object_prompt,
            )
        else:
            return {"hit": False}

    def _handle_whole_cache_hit(self, search_result, similarity_score):
        """处理完整提示的缓存命中"""
        cache_id = search_result.id
        cached_prompt = search_result.payload.get("prompt", "")
        saved_steps = search_result.payload.get("saved_steps", [])

        self._update_cache_access(cache_id)

        skip_step = self._determine_skip_step(similarity_score, saved_steps)
        if skip_step == 0:
            return {"hit": False}

        try:
            latent_state = None

            latent_key = f"{cache_id}_step{skip_step}"
            get_result = self.kv_cache_store.get(latent_key)
            future_get = get_result.unwrap().wait()
            logger.info(f"🚐 Getting {latent_key} from KVCacheStore...")

            if future_get.error() is None:
                latent_bytes = future_get.success()
                latent_bytes = latent_bytes.bytes().success()
                latent_state = pickle.loads(latent_bytes)
                logger.info(f"😊 Loaded latent state from KVCacheStore: {latent_key}")
            else:
                logger.info(
                    f"😥 Failed to load latent state from KVCacheStore: {future_get.error() if future_get.error() else 'Not found'}"
                )
                return {"hit": False}

            if latent_state is not None:
                device = self.device if hasattr(self, "device") else "cuda"
                if (
                    isinstance(latent_state, torch.Tensor)
                    and latent_state.device.type != device
                ):
                    latent_state = latent_state.to(device)
                return {
                    "hit": True,
                    "skip_step": skip_step,
                    "latent_state": latent_state,
                    "similarity": similarity_score,
                    "cached_prompt": cached_prompt,
                    "cache_type": "whole",
                }
        except Exception as e:
            logger.info(f"Error loading latent state: {e}")
            return {"hit": False}

        return {"hit": False}

    def _handle_decomposed_cache_hit(
        self,
        background_result,
        object_result,
        sim_background,
        sim_object,
        background_prompt,
        object_prompt,
    ):
        """处理分解缓存的命中"""
        bg_cache_id = background_result.id
        obj_cache_id = object_result.id

        bg_saved_steps = background_result.payload.get("saved_steps", [])
        obj_saved_steps = object_result.payload.get("saved_steps", [])

        self._update_cache_access(bg_cache_id)
        self._update_cache_access(obj_cache_id)

        common_steps = set(bg_saved_steps) & set(obj_saved_steps) & set(self.key_steps)
        if not common_steps:
            return {"hit": False}

        min_similarity = min(sim_background, sim_object)
        skip_step = self._determine_skip_step(min_similarity, list(common_steps))
        if skip_step == 0:
            return {"hit": False}

        try:
            bg_latent = None
            obj_latent = None
            bg_mask = None
            obj_mask = None

            bg_latent_key = f"{bg_cache_id}_step{skip_step}"
            get_result = self.kv_cache_store.get(bg_latent_key)
            future_get = get_result.unwrap().wait()

            if future_get.error() is None:
                bg_latent_bytes = future_get.success()
                bg_latent_bytes = bg_latent_bytes.bytes().success()
                bg_latent = pickle.loads(bg_latent_bytes)
                logger.info(
                    f"😊 Loaded background latent state from KVCacheStore: {bg_latent_key}"
                )
            else:
                logger.info(
                    f"😥 Failed to load background latent state from KVCacheStore: {future_get.error() if future_get.error() else 'Not found'}"
                )
                return {"hit": False}

            bg_mask_key = f"{bg_cache_id}_background_mask"
            get_result = self.kv_cache_store.get(bg_mask_key)
            future_get = get_result.unwrap().wait()

            if future_get.error() is None:
                bg_mask_bytes = future_get.success()
                bg_mask_bytes = bg_mask_bytes.bytes().success()
                bg_mask = pickle.loads(bg_mask_bytes)
                logger.info(f"😊 Loaded background mask from KVCacheStore: {bg_mask_key}")
            else:
                logger.info(
                    f"😥 Failed to load background mask from KVCacheStore: {future_get.error() if future_get.error() else 'Not found'}"
                )
                return {"hit": False}

            obj_latent_key = f"{obj_cache_id}_step{skip_step}"
            get_result = self.kv_cache_store.get(obj_latent_key)
            future_get = get_result.unwrap().wait()

            if future_get.error() is None:
                obj_latent_bytes = future_get.success()
                obj_latent_bytes = obj_latent_bytes.bytes().success()
                obj_latent = pickle.loads(obj_latent_bytes)
                logger.info(f"😊 Loaded object latent state from KVCacheStore: {obj_latent_key}")
            else:
                logger.info(
                    f"😥 Failed to load object latent state from KVCacheStore: {future_get.error() if future_get.error() else 'Not found'}"
                )
                return {"hit": False}

            obj_mask_key = f"{obj_cache_id}_object_mask"
            get_result = self.kv_cache_store.get(obj_mask_key)
            future_get = get_result.unwrap().wait()

            if future_get.error() is None:
                obj_mask_bytes = future_get.success()
                obj_mask_bytes = obj_mask_bytes.bytes().success()
                obj_mask = pickle.loads(obj_mask_bytes)
                logger.info(f"😊 Loaded object mask from KVCacheStore: {obj_mask_key}")
            else:
                logger.info(
                    f"😥 Failed to load object mask from KVCacheStore: {future_get.error() if future_get.error() else 'Not found'}"
                )
                return {"hit": False}

            if all(x is not None for x in [bg_latent, obj_latent, bg_mask, obj_mask]):
                device = self.device if hasattr(self, "device") else "cuda"

                if (
                    isinstance(bg_latent, torch.Tensor)
                    and bg_latent.device.type != device
                ):
                    bg_latent = torch.from_numpy(bg_latent).to(device)
                if (
                    isinstance(obj_latent, torch.Tensor)
                    and obj_latent.device.type != device
                ):
                    obj_latent = torch.from_numpy(obj_latent).to(device)

                combined_latent = self._combine_latent_states(
                    bg_latent, obj_latent, bg_mask, obj_mask
                )

                return {
                    "hit": True,
                    "skip_step": skip_step,
                    "latent_state": combined_latent,
                    "similarity": min(sim_background, sim_object),
                    "cached_prompt": f"Background: {background_result.payload.get('prompt', '')} + Object: {object_result.payload.get('prompt', '')}",
                    "cache_type": "decomposed",
                }
        except Exception as e:
            logger.info(f"Error loading decomposed cache: {e}")
            return {"hit": False}

        return {"hit": False}

    def _visualize_combine_latent_states(
        self, bg_mask: np.ndarray, obj_mask: np.ndarray
    ):
        """可视化组合latent states时的mask"""
        try:

            vis_dir = "combine_latent_visualizations"
            os.makedirs(vis_dir, exist_ok=True)

            bg_mask_vis = np.clip(bg_mask, 0, 1).astype(np.float32)
            obj_mask_vis = np.clip(obj_mask, 0, 1).astype(np.float32)

            bg_mask_uint8 = (bg_mask_vis * 255).astype(np.uint8)
            obj_mask_uint8 = (obj_mask_vis * 255).astype(np.uint8)

            bg_mask_path = os.path.join(vis_dir, f"bg_mask.png")
            obj_mask_path = os.path.join(vis_dir, f"obj_mask.png")
            cv2.imwrite(bg_mask_path, bg_mask_uint8)
            cv2.imwrite(obj_mask_path, obj_mask_uint8)

            h, w = bg_mask_vis.shape

            bg_vis = np.zeros((h, w, 3), dtype=np.uint8)
            bg_vis[:, :, 0] = (bg_mask_vis * 255).astype(np.uint8)

            obj_vis = np.zeros((h, w, 3), dtype=np.uint8)
            obj_vis[:, :, 2] = (obj_mask_vis * 255).astype(np.uint8)

            combined_vis = np.zeros((h, w, 3), dtype=np.uint8)
            combined_vis[:, :, 0] = (bg_mask_vis * 255).astype(np.uint8)
            combined_vis[:, :, 2] = (obj_mask_vis * 255).astype(np.uint8)

            bg_vis_path = os.path.join(vis_dir, f"bg_visualization.png")
            obj_vis_path = os.path.join(vis_dir, f"obj_visualization.png")
            combined_vis_path = os.path.join(vis_dir, f"combined_visualization.png")

            cv2.imwrite(bg_vis_path, bg_vis)
            cv2.imwrite(obj_vis_path, obj_vis)
            cv2.imwrite(combined_vis_path, combined_vis)

            comparison = np.hstack([bg_vis, obj_vis, combined_vis])
            comparison_path = os.path.join(vis_dir, f"comparison.png")
            cv2.imwrite(comparison_path, comparison)

            logger.info(f"Combine latent states visualization saved:")
            logger.info(f"  Background mask: {bg_mask_path}")
            logger.info(f"  Object mask: {obj_mask_path}")
            logger.info(f"  Background visualization (blue): {bg_vis_path}")
            logger.info(f"  Object visualization (red): {obj_vis_path}")
            logger.info(f"  Combined visualization (blue+red): {combined_vis_path}")
            logger.info(f"  Comparison: {comparison_path}")

        except Exception as e:
            logger.info(f"Error in combine latent states visualization: {e}")

    def _combine_latent_states(
        self,
        bg_latent: torch.Tensor,
        obj_latent: torch.Tensor,
        bg_mask: np.ndarray,
        obj_mask: np.ndarray,
    ) -> torch.Tensor:
        """组合背景和对象的latent states"""
        device = bg_latent.device
        latent_shape = bg_latent.shape
        h, w = latent_shape[-2], latent_shape[-1]

        bg_mask_resized = cv2.resize(bg_mask[0], (w, h)) / 255.0
        obj_mask_resized = cv2.resize(obj_mask[0], (w, h)) / 255.0

        self._visualize_combine_latent_states(bg_mask_resized, obj_mask_resized)

        bg_mask_tensor = torch.from_numpy(bg_mask_resized).float().to(device)
        obj_mask_tensor = torch.from_numpy(obj_mask_resized).float().to(device)

        bg_mask_tensor = bg_mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        obj_mask_tensor = obj_mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        bg_mask_tensor = bg_mask_tensor.expand_as(bg_latent)
        obj_mask_tensor = obj_mask_tensor.expand_as(obj_latent)

        # 组合latent states
        # 背景区域使用背景latent，对象区域使用对象latent
        combined_latent = (
            bg_latent * (1 - obj_mask_tensor) + obj_latent * obj_mask_tensor
        )

        return combined_latent

    def _determine_skip_step(
        self, similarity_score: float, saved_steps: List[int]
    ) -> int:
        """根据相似度决定可以跳过的步骤"""
        if similarity_score >= 0.9 and 25 in saved_steps:
            return 20
        elif similarity_score >= 0.85 and 20 in saved_steps:
            return 15
        elif similarity_score >= 0.8 and 15 in saved_steps:
            return 10
        elif similarity_score >= 0.75 and 10 in saved_steps:
            return 5
        elif 5 in saved_steps:
            return 5
        else:
            return 0

    def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, torch.Tensor],
        timesteps,
        num_frames: int,
        generated_video_frames: Optional[List[np.ndarray]]= None,
        generated_video_tensor: Optional[torch.Tensor] = None,
    ):

        self._check_and_clean_cache()

        decomposed = self.decompose_prompt_with_llama(prompt)
        background_prompt = decomposed["background"]
        object_prompt = decomposed["object"]

        cache_id = str(uuid.uuid4())

        whole_embedding = self.encode_prompt(prompt)
        background_embedding = self.encode_prompt(background_prompt)
        object_embedding = self.encode_prompt(object_prompt)

        masks = None
        # if generated_video_frames is not None and object_prompt:
        #     masks = self.segment_video_with_langsam(
        #         generated_video_frames, object_prompt
        #     )

        saved_steps = []
        total_size_mb = 0
        save_success = True

        if latent_states_dict:
            for step, latent in latent_states_dict.items():
                if step in self.key_steps:
                    try:
                        latent_key = f"{cache_id}_step{step}"
                        latent_bytes = pickle.dumps(latent)

                        put_result = self.kv_cache_store.put(latent_key, latent_bytes)
                        future_put = put_result.unwrap().wait()
                        total_size_mb += len(latent_bytes) / (1024 * 1024)
                        if future_put.error() is not None:
                            logger.info(
                                f"😥 Failed to save step {step} to KVCacheStore: {len(latent_bytes)/1024:.2f}KB {future_put.error()}"
                            )
                            save_success = False
                            break

                        saved_steps.append(step)
                        logger.info(
                            f"😊 Saved step {step} to KVCacheStore: {len(latent_bytes)/1024:.2f}KB,latent_key: {latent_key}"
                        )

                    except Exception as e:
                        logger.info(f"😥 Failed to save step {step} {len(latent_bytes)/1024:.2f}KB,: {e}")
                        save_success = False
                        break

        if masks is not None and save_success:
            try:
                obj_mask_key = f"{cache_id}_object_mask"
                obj_mask_bytes = pickle.dumps(np.array(masks))

                put_result = self.kv_cache_store.put(obj_mask_key, obj_mask_bytes)
                future_put = put_result.unwrap().wait()

                if future_put.error() is not None:
                    logger.info(
                        f"Failed to save object mask to KVCacheStore: {future_put.error()} {len(obj_mask_bytes)/1024:.2f}"
                    )
                else:
                    logger.info(
                        f"Saved object mask to KVCacheStore,latent_key: {obj_mask_key} {len(obj_mask_bytes)/1024:.2f}"
                    )

                bg_masks = [255 - mask for mask in masks]
                bg_mask_key = f"{cache_id}_background_mask"
                bg_mask_bytes = pickle.dumps(np.array(bg_masks))

                put_result = self.kv_cache_store.put(bg_mask_key, bg_mask_bytes)
                future_put = put_result.unwrap().wait()

                if future_put.error() is not None:
                    logger.info(
                        f"Failed to save background mask to KVCacheStore: {future_put.error()} {future_put.error()} {len(bg_mask_bytes)/1024:.2f}"
                    )
                else:
                    logger.info(
                        f"Saved background mask to KVCacheStore,latent_key: {bg_mask_key} {len(bg_mask_bytes)/1024:.2f}"
                    )

            except Exception as e:
                logger.info(f"Failed to save masks: {e}")


        video_meta = {
            "has_video": False,
            "video_blob_key": None,
            "video_encoding": None,
            "video_sha256": None,
            "video_size_bytes": None,
            "fps": None,
            "resolution": None,
            "video_duration_seconds": None,
        }

        if generated_video_tensor is not None and save_success:
            try:
                tensor_blob_key = f"{cache_id}_video_tensor"
                tensor_to_save = generated_video_tensor.detach()
                if tensor_to_save.device.type != "cpu":
                    tensor_to_save = tensor_to_save.cpu()
                import io
                buf = io.BytesIO()
                torch.save(tensor_to_save, buf)
                tensor_bytes = buf.getvalue()
                video_size_bytes = len(tensor_bytes)
                video_sha256 = hashlib.sha256(tensor_bytes).hexdigest()

                put_result = self.kv_cache_store.put(tensor_blob_key, tensor_bytes)
                future_put = put_result.unwrap().wait()

                if future_put.error() is not None:
                    logger.info(
                        f"Failed to save tensor to KVCacheStore: {future_put.error()} {video_size_bytes/1024:.2f}KB"
                    )
                else:
                    total_size_mb += video_size_bytes / (1024 * 1024)
                    try:
                        h = int(tensor_to_save.shape[1])
                        w = int(tensor_to_save.shape[2])
                        resolution = (w, h)
                    except Exception:
                        resolution = None

                    video_meta.update(
                        {
                            "has_video": True,
                            "video_blob_key": tensor_blob_key,
                            "video_encoding": "torch_tensor",
                            "video_sha256": video_sha256,
                            "video_size_bytes": video_size_bytes,
                            "resolution": resolution,
                        }
                    )
                    logger.info(
                        f"Saved tensor video to KVCacheStore: {video_size_bytes/1024:.2f}KB, key: {tensor_blob_key}"
                    )
            except Exception as e:
                logger.info(f"Failed to save tensor to KVCacheStore: {e}")

        if saved_steps and save_success:

            points_data = [
                (self.collection_name_whole, whole_embedding, prompt),
                (
                    self.collection_name_background,
                    background_embedding,
                    background_prompt,
                ),
                (self.collection_name_object, object_embedding, object_prompt),
            ]

            for collection_name, embedding, prompt_text in points_data:
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=cache_id,
                            vector=embedding.tolist(),
                            payload={
                                "prompt": prompt_text,
                                "saved_steps": saved_steps,
                                "num_frames": num_frames,
                                "original_prompt": prompt,
                                "has_masks": masks is not None,
                            },
                        )
                    ],
                )


        has_video = video_meta.get("has_video", False)
        if has_video and save_success:
            try:
                self.client.upsert(
                    collection_name=self.collection_name_video,
                    points=[
                        PointStruct(
                            id=cache_id,
                            vector=whole_embedding.tolist(),
                            payload={
                                "prompt": prompt,
                                "num_frames": num_frames,
                                "saved_steps": saved_steps,
                                "has_video": True,
                                "video_blob_key": video_meta["video_blob_key"],
                                "video_encoding": video_meta["video_encoding"],
                                "video_sha256": video_meta["video_sha256"],
                                "video_size_bytes": video_meta["video_size_bytes"],
                                "resolution": video_meta["resolution"],
                            },
                        )
                    ],
                )
                logger.info("Indexed video entry into video collection")
            except Exception as e:
                logger.info(f"Failed to upsert into video collection: {e}")


        if (saved_steps or has_video) and save_success:
            self.cache_metadata[cache_id] = {
                "prompt": prompt,
                "background_prompt": background_prompt,
                "object_prompt": object_prompt,
                "saved_steps": len(saved_steps),
                "capacity": num_frames,
                "access_count": 0,
                "last_access_time": time.time(),
                "size_mb": total_size_mb,
                "has_masks": masks is not None,
                "has_video": has_video,
                "video_blob_key": video_meta.get("video_blob_key"),
                "video_encoding": video_meta.get("video_encoding"),
                "video_sha256": video_meta.get("video_sha256"),
                "video_size_bytes": video_meta.get("video_size_bytes"),
                "resolution": video_meta.get("resolution"),
            }

            self._save_cache_metadata()

            logger.info(f"Successfully cached prompt: {prompt}")
            logger.info(f"Background: {background_prompt}")
            logger.info(f"Object: {object_prompt}")
            logger.info(f"Saved steps: {saved_steps}")
            logger.info(f"Has video: {has_video}")
            logger.info(f"Total cache size: {total_size_mb:.2f} MB")
        else:
            logger.info(f"Failed to cache prompt due to storage errors")

    def _check_and_clean_cache(self):
        """使用LRBU策略检查并清理缓存，每个集合独立驱逐"""
        total_size_mb = sum(meta["size_mb"] for meta in self.cache_metadata.values())
        total_size_gb = total_size_mb / 1024

        if total_size_gb > self.max_cache_size_gb:
            logger.info(
                f"Cache size ({total_size_gb:.2f}GB) exceeds limit ({self.max_cache_size_gb}GB)"
            )

            collection_configs = [
                {"name": self.collection_name_whole, "type": "whole", "weight": 1.0},
                {
                    "name": self.collection_name_background,
                    "type": "background",
                    "weight": 0.8,
                },
                {"name": self.collection_name_object, "type": "object", "weight": 0.8},
            ]

            all_candidates = []
            current_time = time.time()

            for config in collection_configs:
                collection_name = config["name"]
                cache_type = config["type"]
                weight = config["weight"]

                try:
                    search_result = self.client.scroll(
                        collection_name=collection_name, limit=10000, with_payload=True
                    )[0]

                    for point in search_result:
                        cache_id = point.id
                        if cache_id in self.cache_metadata:
                            meta = self.cache_metadata[cache_id]

                            f_i = meta["access_count"] + 1
                            step_i = meta["saved_steps"]
                            capacity_i = meta["capacity"]
                            duration_i = current_time - meta["last_access_time"] + 1

                            priority = (f_i * step_i * weight) / (
                                capacity_i * duration_i
                            )

                            all_candidates.append(
                                {
                                    "priority": priority,
                                    "cache_id": cache_id,
                                    "collection": collection_name,
                                    "type": cache_type,
                                    "size_mb": meta["size_mb"],
                                    "storage_type": meta.get("storage_type", "local"),
                                }
                            )

                except Exception as e:
                    logger.info(f"Error accessing collection {collection_name}: {e}")
                    continue

            all_candidates.sort(key=lambda x: x["priority"])

            while total_size_gb > self.max_cache_size_gb * 0.9 and all_candidates:
                candidate = all_candidates.pop(0)
                cache_id = candidate["cache_id"]
                collection_name = candidate["collection"]
                cache_type = candidate["type"]
                storage_type = candidate["storage_type"]

                try:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=[cache_id]),
                    )
                    logger.info(f"Deleted cache {cache_id} from {cache_type} collection")
                except Exception as e:
                    logger.info(f"Failed to delete from {collection_name}: {e}")

                still_exists_in_other_collections = False
                for other_config in collection_configs:
                    if other_config["name"] != collection_name:
                        try:
                            retrieve_result = self.client.retrieve(
                                collection_name=other_config["name"], ids=[cache_id]
                            )
                            if retrieve_result:
                                still_exists_in_other_collections = True
                                break
                        except:
                            continue

                if not still_exists_in_other_collections:
                    deleted_size = 0

                    for step in self.key_steps:
                        latent_key = f"{cache_id}_step{step}"
                        try:
                            is_exist_result = self.kv_cache_store.is_exist(latent_key)
                            future_is_exist = is_exist_result.unwrap().wait()

                            if (
                                future_is_exist.error() is None
                                and future_is_exist.value()
                            ):

                                get_size_result = self.kv_cache_store.get_size(
                                    latent_key
                                )
                                future_get_size = get_size_result.unwrap().wait()

                                if future_get_size.error() is None:
                                    file_size = future_get_size.value() / (1024 * 1024)
                                else:
                                    file_size = 0

                                remove_result = self.kv_cache_store.remove(latent_key)
                                future_remove = remove_result.unwrap().wait()

                                if future_remove.error() is None:
                                    deleted_size += file_size
                                    logger.info(
                                        f"Deleted latent key from KVCacheStore: {latent_key}"
                                    )
                                else:
                                    logger.info(
                                        f"Failed to delete latent key from KVCacheStore: {latent_key}, error: {future_remove.error()}"
                                    )
                        except Exception as e:
                            logger.info(
                                f"Failed to delete latent key from KVCacheStore: {latent_key}, error: {e}"
                            )

                    for mask_type in ["object", "background"]:
                        mask_key = f"{cache_id}_{mask_type}_mask"
                        try:

                            is_exist_result = self.kv_cache_store.is_exist(mask_key)
                            future_is_exist = is_exist_result.unwrap().wait()

                            if (
                                future_is_exist.error() is None
                                and future_is_exist.value()
                            ):

                                remove_result = self.kv_cache_store.remove(mask_key)
                                future_remove = remove_result.unwrap().wait()

                                if future_remove.error() is None:
                                    logger.info(
                                        f"Deleted mask key from KVCacheStore: {mask_key}"
                                    )
                                else:
                                    logger.info(
                                        f"Failed to delete mask key from KVCacheStore: {mask_key}, error: {future_remove.error()}"
                                    )
                        except Exception as e:
                            logger.info(
                                f"Failed to delete mask key from KVCacheStore: {mask_key}, error: {e}"
                            )

                    total_size_gb -= deleted_size / 1024
                    if cache_id in self.cache_metadata:
                        del self.cache_metadata[cache_id]

                    logger.info(
                        f"LRBU evicted cache: {cache_id} from all collections (priority: {candidate['priority']:.6f})"
                    )
                else:
                    logger.info(
                        f"Cache {cache_id} still exists in other collections, keeping physical files"
                    )

                all_candidates = [
                    c for c in all_candidates if c["cache_id"] != cache_id
                ]

            self._save_cache_metadata()

    def _update_cache_access(self, cache_id):
        """更新缓存访问统计"""
        if cache_id in self.cache_metadata:
            self.cache_metadata[cache_id]["access_count"] += 1
            self.cache_metadata[cache_id]["last_access_time"] = time.time()
            self._save_cache_metadata()

    def _save_cache_metadata(self):
        """保存缓存元数据到文件"""
        metadata_path = os.path.join(self.cache_dir, "cache_metadata.pkl")
        logger.info(f"Saving metadata to: {metadata_path}")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.cache_metadata, f)

    def _load_cache_metadata(self):
        """从文件加载缓存元数据"""
        metadata_path = os.path.join(self.cache_dir, "cache_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                return pickle.load(f)
        return {}

    def get_storage_stats(self):
        """获取存储统计信息"""
        stats = {
            "storage_type": "Enhanced Local with Object/Background Separation",
            "storage_path": self.latent_storage_path,
            "mask_path": self.mask_storage_path,
            "total_caches": len(self.cache_metadata),
            "total_size_mb": sum(
                meta["size_mb"] for meta in self.cache_metadata.values()
            ),
            "caches_with_masks": sum(
                1
                for meta in self.cache_metadata.values()
                if meta.get("has_masks", False)
            ),
            "collections": {
                "whole_prompts": self.collection_name_whole,
                "background_prompts": self.collection_name_background,
                "object_prompts": self.collection_name_object,
            },
        }

        return stats


    def search_video_collection(self, prompt: str, limit: int = 1, min_score: Optional[float] = None):
        """在 video 集合中根据 prompt 检索相似视频。
        返回列表，每项包含 {"id", "score", "payload"}。
        """
        try:
            whole_embedding = self.encode_prompt(prompt)
            search_params = models.SearchParams(hnsw_ef=256, exact=False)

            results = self.client.search(
                collection_name=self.collection_name_video,
                query_vector=whole_embedding.tolist(),
                limit=limit,
                search_params=search_params,
            )

            out = []
            for r in results:
                if min_score is not None and r.score < min_score:
                    continue
                out.append({"id": r.id, "score": r.score, "payload": dict(r.payload or {})})
            return out
        except Exception as e:
            logger.info(f"search_video_collection error: {e}")
            return []

    def get_saved_video_by_prompt(self, prompt: str, limit: int = 1, min_score: Optional[float] = None) -> Dict[str, Union[bool, Optional[Union[str, int]], Optional[float], Optional[torch.Tensor], Optional[bytes], Dict]]:
        """按提示检索 video 集合并返回已保存的视频（仅支持张量）。
        返回字典: {hit, id, score, video_tensor, video_bytes, payload}
        - 仅当编码为 "torch_tensor" 时返回 video_tensor；其他情况视为未命中
        """
        try:
            min_score = min_score if min_score is not None else self.video_similarity_threshold
            results = self.search_video_collection(prompt, limit=limit, min_score=min_score)
            if not results:
                return {"hit": False, "id": None, "score": None, "video_tensor": None, "video_bytes": None, "payload": {}}

            best = results[0]
            payload = best.get("payload", {})
            video_blob_key = payload.get("video_blob_key")
            if not video_blob_key:
                return {"hit": False, "id": best.get("id"), "score": float(best.get("score", 0.0)), "video_tensor": None, "video_bytes": None, "payload": payload}

            video_encoding = payload.get("video_encoding")
            video_tensor = None
            if video_encoding == "torch_tensor":
                video_tensor = self._get_video_tensor_from_kv(video_blob_key)
                if video_tensor is None:
                    return {"hit": False, "id": best.get("id"), "score": float(best.get("score", 0.0)), "video_tensor": None, "video_bytes": None, "payload": payload}
            else:
                # 仅支持 torch_tensor，其它编码视为未命中
                return {"hit": False, "id": best.get("id"), "score": float(best.get("score", 0.0)), "video_tensor": None, "video_bytes": None, "payload": payload}

            cache_id = best.get("id")
            if cache_id:
                self._update_cache_access(cache_id)

            return {
                "hit": True,
                "id": cache_id,
                "score": float(best.get("score", 0.0)),
                "video_tensor": video_tensor,
                "video_bytes": None,
                "payload": payload,
            }
        except Exception as e:
            logger.info(f"get_saved_video_by_prompt error: {e}")
            return {"hit": False, "id": None, "score": None, "video_tensor": None, "video_bytes": None, "payload": {}}


    def _get_video_tensor_from_kv(self, video_blob_key: str) -> Optional[torch.Tensor]:
        """从 KVStore 获取并反序列化保存的 torch 视频张量。"""
        try:
            if self.kv_cache_store is None:
                logger.info("KVCacheStore not initialized")
                return None
            get_result = self.kv_cache_store.get(video_blob_key)
            future_get = get_result.unwrap().wait()
            if future_get.error() is None:
                blob = future_get.success()
                data = blob.bytes().success()
                import io
                buf = io.BytesIO(data)
                tensor = torch.load(buf, map_location="cpu")
                return tensor
            else:
                logger.info(f"KV get error for {video_blob_key}: {future_get.error()}")
                return None
        except Exception as e:
            logger.info(f"_get_video_tensor_from_kv exception: {e}")
            return None
