from teleai_pipe.pipelines.teleai_video_v1.pipeline_diffusers import VideoPipeline
import torch
from diffusers.utils import export_to_video
from typing import List, Dict
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import os
import glob
import pickle
from teleai_pipe.utils.vision import (
    GeneratePoseControlImages,
    GenerateBoxControlImages,
    create_4grid_video,
)
import json
from tqdm import tqdm
import random
import imageio
import numpy as np


def prepare_reference_images(
    width: int, height: int, num_frames: int, ref_frames: Dict[int, str]
) -> List[Image.Image]:
    """准备参考图像"""
    ref_images = [
        Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(num_frames)
    ]
    for frame_idx, img_path in ref_frames.items():
        if frame_idx >= num_frames:
            raise ValueError(f"参考帧索引 {frame_idx} 超过总帧数 {num_frames}")

        original_img = Image.open(img_path).convert("RGB")
        resized_img = F.resize(
            original_img, (height, width), InterpolationMode.BILINEAR
        )
        ref_images[frame_idx] = resized_img
    return ref_images


base_model_path = "/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
transformer_model_path = "/nvfile-heatstorage/cjf/dev/json_dataset/vast/work_dirs/hunyuanvideo/hunyuanvideo_i2v_condition_20250402_new_dataset/models/checkpoint_epoch_3_step_7400/transformer"
width = 1280
height = 720
num_frames = 45
num_inference_steps = 50
output_dir = "work_dirs/output/"
os.makedirs(output_dir, exist_ok=True)
pipeline = VideoPipeline.from_pretrained(
    base_model_path,
    transformer_model_path=transformer_model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipeline.vae.enable_tiling()


def inference_from_json_dataset(idxs, data):
    for idx in tqdm(idxs):
        clip = data["clips"][idx]
        prompt = clip["caption"]["dense_caption"][0]
        print(f"prompt: {prompt}")
        data_dict = {}
        try:
            with open(clip["mesh"], "rb") as file:
                mesh = pickle.load(file)
                data_dict["joints2d_nonparam"] = mesh["joints2d_nonparam"]
                data_dict["joint_uncertainties"] = mesh["joint_uncertainties"]

        except Exception as e:
            print(f"Failed to load mesh: {e}")

        try:
            with open(clip["bbox"], "r") as file:
                content = file.read()
                bbox = json.loads(content)
                data_dict["bbox"] = bbox
        except Exception as e:
            print(f"Failed to load bbox: {e}")

        data_dict["video_height"] = clip["height"]
        data_dict["video_width"] = clip["width"]
        data_dict["sample_indexes"] = list(range(num_frames))

        pose_generator = GeneratePoseControlImages()
        box_generator = GenerateBoxControlImages()
        pose_images = pose_generator(data_dict)
        box_images = box_generator(data_dict)

        reader = imageio.get_reader(
            os.path.join(data["clip_data_root"], clip["file_path"])
        )
        ref_frames = [reader.get_data(i) for i in data_dict["sample_indexes"]]
        ref_images = [Image.fromarray(frame) for frame in ref_frames]

        pose_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in pose_images
        ]
        box_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in box_images
        ]
        ref_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in ref_images
        ]

        # 接口,用于检查pose和box图像是否有效
        if not pose_generator.if_valid:
            print("no valid pose images")
        if not box_generator.if_valid:
            print("no valid box images")

        video = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=42,
            model_type="i2vhy",
            ref_images=ref_images,
            pose_images=pose_images,
            box_images=box_images,
            guidance_scale=1.0,
            teacache_l1_th=0,
        ).frames[0]

        export_to_video(video, output_dir + f"{idx}_cfg.mp4", fps=15)
        # 将pose,box,ref和生成的视频合并为一个视频进行展示
        combined_frames = create_4grid_video(
            [pose_images, box_images, ref_images, video]
        )
        imageio.mimsave(
            output_dir + f"{idx}_combined.mp4",
            [np.array(frame) for frame in combined_frames],
            fps=16,
        )


json_path = f"{os.path.dirname(__file__)}/../data/test_s2v.json"
with open(json_path, "r") as file:
    data = json.load(file)

inference_from_json_dataset([0], data)
