from teleai_pipe.pipelines.wan_video import WanVideoPipeline as VideoPipeline
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.utils.utils import load_state_dict

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
import torch.nn as nn


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


model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16,  # Image Encoder is loaded with float16
)

dit_path = [
    os.path.join(model_root, f)
    for f in os.listdir(model_root)
    if f.endswith(".safetensors")
]
model_manager.load_models(
    [
        dit_path,
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_root}/Wan2.1_VAE.pth",
    ],
    # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
# lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504011_lora/models/checkpoint_epoch_2_step_6400/transformer/transformer_lora.pt"
lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504015_lora/models/checkpoint_epoch_2_step_2900/transformer/transformer_lora.pt"
# "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504015_lora/models/checkpoint_epoch_1_step_1300/transformer/transformer_lora.pt"
lora_state_dict = load_state_dict(lora_model_path)
model_manager.load_lora("", lora_state_dict, lora_alpha=1.0)

pipe = VideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda",
    compile=False,
)
pipe.dit.patch_embedding = nn.Conv3d(
    in_channels=52,
    out_channels=5120,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
    dtype=torch.bfloat16,
)
pipe.dit.load_state_dict(lora_state_dict, strict=False)
pipe.enable_compile_dit()
pipe.enable_cpu_offload()

width = 1280
height = 720
num_frames = 45
num_inference_steps = 50
output_dir = "work_dirs/output/"
os.makedirs(output_dir, exist_ok=True)


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

        video = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=5.0,
            seed=0,
            tiled=True,
            tile_size=(34, 34),
            tile_stride=(18, 16),
            input_image=ref_images[0],
            cn_images=pose_images,
            model_type="Wan2.1-I2V-14B-720P",
            tea_cache_l1_thresh=0.15,
            attn_impl="cudnn",
        )

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


# json_path = f"{os.path.dirname(__file__)}/../data/test_s2v.json"
json_path = "/nvfile-heatstorage/cjf/share/test_keling/ticao.json"
with open(json_path, "r") as file:
    data = json.load(file)

inference_from_json_dataset([-1], data)
