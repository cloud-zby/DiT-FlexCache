import os
import re
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.pipelines.teleai_t2s_v1 import TeleAIT2SPipeline


class MotionDatabase:
    """
    A class for matching user inputs with motion descriptions and finding the most similar motions.
    """

    def __init__(
        self,
        text_dir,
        motion_dir,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        threshold=0.6,
        cache_file="motion_matcher_cache.pkl",
    ):
        """
        Initialize the MotionTextMatcher.

        Args:
            text_dir (str): Directory containing text description files
            motion_dir (str): Directory containing motion data files
            model_name (str): Name of the sentence transformer model to use
            threshold (float): Similarity threshold for determining strong matches
            cache_file (str): Path to cache file for storing/loading index data
        """
        self.text_dir = text_dir
        self.motion_dir = motion_dir
        self.threshold = threshold
        self.cache_file = cache_file

        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)

        # Cache for text embeddings
        self.text_cache = {}

        # Try to load cache if it exists
        self.load_cache()

        if len(self.text_cache) == 0:
            num_indexed = self.build_index()
            print(f"已成功索引 {num_indexed} 个文件")
        else:
            print(f"已从缓存加载 {len(self.text_cache)} 个索引项")

    def preprocess_text_file(self, file_path):
        """
        Preprocess a text file to extract clean descriptions.

        Args:
            file_path (str): Path to the text file

        Returns:
            list: List of clean descriptions
        """
        descriptions = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                if line.strip():
                    # Remove scoring part (after #)
                    clean_text = line.split("#")[0].strip()

                    # Extract words without POS tags
                    pure_text = " ".join(
                        [word.split("/")[0] for word in clean_text.split()]
                    )

                    # Clean up extra spaces
                    pure_text = re.sub(r"\s+", " ", pure_text).strip()

                    if pure_text:
                        descriptions.append(pure_text)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

        return descriptions

    def calculate_similarity(self, user_input, text_descriptions):
        """
        Calculate similarity between user input and text descriptions.

        Args:
            user_input (str): User input text
            text_descriptions (list): List of text descriptions

        Returns:
            numpy.ndarray: Array of similarity scores
        """
        # Encode user input
        user_embedding = self.model.encode([user_input])[0]

        # Encode text descriptions
        text_embeddings = self.model.encode(text_descriptions)

        # Calculate cosine similarity
        similarities = cosine_similarity([user_embedding], text_embeddings)[0]

        return similarities

    def save_cache(self):
        """
        Save the text cache to a file for faster loading in future runs.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.text_cache, f)
            print(f"Cache saved to {self.cache_file}")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def load_cache(self):
        """
        Load the text cache from a file if it exists.

        Returns:
            bool: True if cache was loaded, False otherwise
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.text_cache = pickle.load(f)
                print(
                    f"Cache loaded from {self.cache_file} with {len(self.text_cache)} entries"
                )
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.text_cache = {}
                return False
        return False

    def build_index(self):
        """
        Build an index of all text descriptions for faster searching.

        Returns:
            int: Number of indexed files
        """
        # If cache is already loaded, return its size
        if self.text_cache:
            return len(self.text_cache)

        count = 0

        for filename in tqdm(os.listdir(self.text_dir)):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.text_dir, filename)

                # Preprocess text descriptions
                descriptions = self.preprocess_text_file(file_path)

                if descriptions:
                    # Calculate embeddings for all descriptions
                    embeddings = self.model.encode(descriptions)

                    # Store in cache
                    self.text_cache[filename] = {
                        "descriptions": descriptions,
                        "embeddings": embeddings,
                        "image_file": str(
                            list(
                                (Path(self.text_dir).parent / "first").glob(
                                    f"{filename.replace('.txt', '')}.*"
                                )
                            )[0]
                        ),
                    }
                    count += 1
            # if count>=1000:
            #     break

        # Save the cache after building
        self.save_cache()

        return count

    def load_motion_data(self, motion_file):
        """
        Load motion data from NPZ file.

        Args:
            motion_file (str): Path to motion data file

        Returns:
            dict: Motion data
        """
        try:
            motion_path = os.path.join(self.motion_dir, motion_file)
            # data = np.load(motion_path)
            # return data
            with open(motion_path, "rb") as f:
                data = pickle.load(f)
                return data["poses"]  # {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Error loading motion file {motion_file}: {e}")
            return None

    def retrieval(self, user_input, top_k=1):
        """
        Find motion descriptions that match the user input.

        Args:
            user_input (str): User input text
            top_k (int): Number of top matches to return

        Returns:
            dict: Result containing matching information
        """
        motions = []

        # Encode user input
        user_embedding = self.model.encode([user_input])[0]

        # If cache is empty, build it
        if not self.text_cache:
            self.build_index()

        # Iterate through cached text descriptions
        for filename, data in self.text_cache.items():
            descriptions = data["descriptions"]
            embeddings = data["embeddings"]

            # Calculate similarities using cached embeddings
            similarities = cosine_similarity([user_embedding], embeddings)[0]

            # Get max similarity and index
            max_sim = np.max(similarities)
            max_idx = np.argmax(similarities)

            # Get corresponding motion filename
            motion_filename = os.path.splitext(filename)[0] + ".pkl"
            # motion_path = os.path.join(self.motion_dir, motion_filename)

            motions.append(
                {
                    "text_file": filename,
                    "motion_file": motion_filename,
                    "similarity": float(
                        max_sim
                    ),  # Convert to Python float for JSON serialization
                    "best_description": descriptions[max_idx],
                    "all_descriptions": descriptions,
                    "image_file": data["image_file"],
                }
            )

        # Sort by similarity
        motions.sort(key=lambda x: x["similarity"], reverse=True)

        # Check if there's a strong match
        if motions and motions[0]["similarity"] >= self.threshold:
            motion_file = motions[0]["motion_file"]
            motion_data = self.load_motion_data(motion_file)
            motions[0]["motion"] = motion_data
            return motions[0]
        else:
            return None


def get_image_size(img_path: str) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(img_path)
    return img.shape[:2][::-1]


def norm_pose(pose: Tensor, img_size: Tuple[int, int] = (1920, 1080)):
    return pose / torch.tensor(img_size, dtype=torch.float32).to(device=pose.device)


def unnorm_pose(pose: Tensor, img_size: Tuple[int, int] = (1920, 1080)):
    return pose * torch.tensor(img_size, dtype=torch.float32).to(device=pose.device)


class TeleAIT2SRetrivalPipeline(BasePipeline):
    def __init__(
        self,
        t2s_vae_path: str,
        t2s_tf_path: str,
        motion_text_path: str,
        motion_path: str,
        motion_embed_model_path: str,
        motion_cache_file: str,
    ) -> None:
        super().__init__()
        # 设置路径
        # 创建匹配器实例
        # matcher = MotionTextMatcher(text_dir, motion_dir, model_dir, threshold=0.7)
        self.motion_database = MotionDatabase(
            motion_text_path,
            motion_path,
            motion_embed_model_path,
            threshold=0.7,
            cache_file=motion_cache_file,
        )

        self.t2s_pipeline = TeleAIT2SPipeline.from_pretrained(
            t2s_vae_path, t2s_tf_path, device="cuda", torch_dtype=torch.float32
        )

    def _interpolate(self, motion: Tensor, n: int) -> Tensor:
        N, _, _ = motion.shape
        inter_frames = []
        for i in range(N - 1):
            inter_frames.append(motion[i])
            cur_frame = motion[i]
            next_frame = motion[i + 1]
            for j in range(1, n + 1):
                alpha = j / (n + 1)
                inter_frame = cur_frame * (1 - alpha) + alpha * next_frame
                inter_frames.append(inter_frame)
        inter_frames.append(motion[-1])
        inter_motion = np.array(inter_frames)
        inter_motion = torch.tensor(inter_motion)
        return inter_motion

    def _shift_motion_by_first_pose(
        self,
        motion: torch.Tensor,
        first_pose: torch.Tensor,
        pelvis_joint_idx: int = 0,
    ) -> torch.Tensor:
        """
        使用 bounding box 来对齐和缩放生成的动作序列，使其首帧与目标 pose 对齐，避免相机抖动。

        参数：
            generated_motion: Tensor，shape [T, J, D]，归一化后的动作序列
            first_pose_gt: Tensor，shape [J, D]，目标首帧 pose（未归一化）
            pelvis_joint_idx: int，骨盆索引
        返回：
            adjusted_motion: Tensor，shape [T, J, D]，后处理后的动作
        """
        # Step 1: 目标首帧的最大边长（用于对比）
        target_bbox_size = first_pose.max(dim=0).values - first_pose.min(dim=0).values
        target_size = target_bbox_size.max()  # scalar

        # Step 2: 所有帧的 bbox 最大边长
        min_coords = motion.amin(dim=1)  # [T, D]
        max_coords = motion.amax(dim=1)  # [T, D]
        bbox_sizes = max_coords - min_coords  # [T, D]
        frame_scales = bbox_sizes.max(dim=1).values  # [T]

        # Step 3: 计算平均缩放比例
        scale_ratios = target_size / frame_scales  # [T]
        scale_ratio = (scale_ratios.sum() - scale_ratios.max() - scale_ratios.min()) / (
            scale_ratios.shape[0] - 2
        )  # scalar

        # Step 4: 缩放
        scaled_motion = motion * scale_ratio

        # Step 5: 对齐 pelvis
        pelvis_gen = scaled_motion[0, pelvis_joint_idx]  # [D]
        pelvis_gt = first_pose[pelvis_joint_idx]  # [D]
        pelvis_offset = pelvis_gt - pelvis_gen  # [D]

        # Step 6: 平移所有帧
        adjusted_motion = scaled_motion + pelvis_offset  # broadcasting

        return adjusted_motion

    def _extract_main_pose(self, motions: list) -> Tensor:
        first_person_poses = []
        for pose in motions:
            if pose.shape[0] == 0:
                continue
            first_person_poses.append(pose[0].tolist())
        motion = torch.tensor(first_person_poses)
        return motion

    def _is_pose_normed(self, motion: Tensor) -> bool:
        return (motion.max() <= 1).item()

    def postprocess(
        self, control: Tensor, gen_data: dict, is_retrieval: bool
    ) -> Tensor:
        if is_retrieval:
            final_motion = self._retrieval_postprocess(control, gen_data)
            gen_data["motion"] = final_motion
        else:
            gen_data["motion"] = self._interpolate(gen_motion, 1)

    def _retrieval_postprocess(self, control: Tensor, retrieval_data: dict) -> Tensor:
        gen_motion = retrieval_data["motion"]
        retrieval_image_size = get_image_size(retrieval_data["image_file"])
        gen_motion = self._extract_main_pose(gen_motion)
        if not self._is_pose_normed(gen_motion):
            gen_motion = norm_pose(gen_motion, retrieval_image_size)
        shifted_motion = self._shift_motion_by_first_pose(gen_motion, control)
        gen_motion = torch.concat([control.unsqueeze(0), shifted_motion])
        gen_motion = unnorm_pose(gen_motion, retrieval_image_size)
        return gen_motion

    def __call__(
        self, prompt, cn_translation, control, num_frames, infer_opt
    ) -> Tuple[bool, dict]:
        is_retrieval = False
        cn_data = self.motion_database.retrieval(cn_translation)
        en_data = self.motion_database.retrieval(prompt)
        data = cn_data if cn_data is not None else en_data
        if data is None:
            gen_motion = self.t2s_pipeline(prompt, control, num_frames, infer_opt)[0]
            data = {"motion": gen_motion}
            self.postprocess(control.cpu().detach(), data, is_retrieval=False)
        else:
            is_retrieval = True
            gen_motion = self.postprocess(
                control.cpu().detach(), data, is_retrieval=True
            )
        return is_retrieval, data

    def align_motion_and_prepend_A(
        self, A: np.ndarray, B: np.ndarray, wa: int, ha: int, wb: int, hb: int
    ) -> list[np.ndarray]:
        """
        将连续motion B的空间位置和大小向首图motion A靠齐，并将转换后的A作为第一帧，
        输出为列表。

        将A和B都归一化到[0, 1]空间，在归一化空间中计算B的第一个帧相对于A的放缩和偏移，
        并将这些变换应用到归一化后的B的所有帧。最后将归一化后的A和对齐后的B转换回
        (wb, hb)空间，并将它们组合成一个列表。

        Args:
            A: 形状为 [1, 24, 2] 的 NumPy 数组，表示首图motion (坐标相对于 wa x ha)。
            B: 形状为 [n, 24, 2] 的 NumPy 数组，表示连续motion (坐标相对于 wb x hb)。
            wa: A 所在的原始空间的宽度。
            ha: A 所在的原始空间的高度。
            wb: B 所在的原始空间的宽度。
            hb: B 所在的原始空间的高度。

        Returns:
            一个包含 n+1 个 NumPy 数组的列表，每个数组形状为 [1, 24, 2]。
            列表的第一个元素是转换到 wb x hb 空间的 A，
            后续元素是对齐后转换到 wb x hb 空间的连续motion的每一帧。
        """
        if A.shape != (1, 24, 2):
            raise ValueError("A 的形状必须是 [1, 24, 2]")
        if B.ndim != 3 or B.shape[1:] != (24, 2):
            raise ValueError("B 的形状必须是 [n, 24, 2]")
        if wa <= 0 or ha <= 0 or wb <= 0 or hb <= 0:
            raise ValueError("宽度和高度必须是正数")

        n_frames = B.shape[0]  # 获取 B 的帧数

        # 1. 将 A 和 B 的坐标归一化到 [0, 1] 空间
        A_norm = A.copy().astype(float)  # 使用 float 类型进行精确计算
        A_norm[:, :, 0] = A_norm[:, :, 0] / wa
        A_norm[:, :, 1] = A_norm[:, :, 1] / ha

        B_norm = B.copy().astype(float)  # 使用 float 类型进行精确计算
        B_norm[:, :, 0] = B_norm[:, :, 0] / wb
        B_norm[:, :, 1] = B_norm[:, :, 1] / hb

        # 现在 A_norm 和 B_norm 都在 [0, 1] 统一空间中

        A0_norm = A_norm[0]  # 提取归一化A的第一个帧，形状 [24, 2]
        B0_norm = B_norm[0]  # 提取归一化B的第一个帧，形状 [24, 2]

        # 2. 在 [0, 1] 空间中计算质心 (Centroid)
        centroid_A_norm = np.mean(A0_norm, axis=0)  # 形状 [2]
        centroid_B_norm = np.mean(B0_norm, axis=0)  # 形状 [2]

        # 3. 在 [0, 1] 空间中计算去质心后的数据 (Centered data)
        A0_norm_centered = A0_norm - centroid_A_norm  # 形状 [24, 2]
        B0_norm_centered = B0_norm - centroid_B_norm  # 形状 [24, 2]

        # 4. 在 [0, 1] 空间中计算尺度 (Scale)
        scale_A_norm_sq = np.sum(A0_norm_centered**2)
        scale_B_norm_sq = np.sum(B0_norm_centered**2)

        alignment_scale_factor_norm = 1.0
        epsilon = 1e-8  # 用于比较浮点数是否接近零

        if scale_B_norm_sq > epsilon:
            if scale_A_norm_sq < epsilon:
                alignment_scale_factor_norm = 0.0
            else:
                alignment_scale_factor_norm = np.sqrt(scale_A_norm_sq / scale_B_norm_sq)

        # 5. 在 [0, 1] 空间中计算偏移 (Offset)
        # 目标是将 B0_norm * alignment_scale_factor_norm + offset ≈ A0_norm
        # 将 B0_norm 的质心经过放缩后，平移到 A0_norm 的质心
        # centroid_B_norm * alignment_scale_factor_norm + offset = centroid_A_norm
        # offset = centroid_A_norm - centroid_B_norm * alignment_scale_factor_norm
        alignment_offset_vector_norm = (
            centroid_A_norm - centroid_B_norm * alignment_scale_factor_norm
        )  # 形状 [2]

        # 6. 将计算出的放缩和偏移应用到 B_norm 的所有帧
        B_norm_aligned = (
            B_norm * alignment_scale_factor_norm + alignment_offset_vector_norm
        )  # 形状 [n, 24, 2]

        # B_norm_aligned 的坐标现在相对于 [0, 1] 空间，并且其第一帧 B_norm_aligned[0]
        # 已经对齐到 A_norm[0] 的位置和尺度

        # 7. 将归一化后的 A (A_norm) 和 对齐后的归一化 B (B_norm_aligned)
        # 转换回 B 的原始空间 (wb x hb)

        # 转换 A
        A_in_B_space = A_norm.copy()
        A_in_B_space[:, :, 0] = A_in_B_space[:, :, 0] * wb
        A_in_B_space[:, :, 1] = A_in_B_space[:, :, 1] * hb
        # A_in_B_space 形状仍是 [1, 24, 2]

        # 转换 B_aligned
        B_aligned_in_B_space = B_norm_aligned.copy()
        B_aligned_in_B_space[:, :, 0] = B_aligned_in_B_space[:, :, 0] * wb
        B_aligned_in_B_space[:, :, 1] = B_aligned_in_B_space[:, :, 1] * hb
        # B_aligned_in_B_space 形状是 [n, 24, 2]

        # 8. 将转换回 B 空间的 A 和 B 的帧组合成一个列表
        output_list = [A_in_B_space]  # 列表的第一个元素是转换后的 A

        # 将 B_aligned_in_B_space 的每一帧作为 [1, 24, 2] 添加到列表中
        for i in range(n_frames):
            output_list.append(
                B_aligned_in_B_space[i : i + 1]
            )  # 使用切片保持 [1, 24, 2] 形状

        return output_list
