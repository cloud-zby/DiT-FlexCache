from teleai_pipe.pipelines.wan_video import WanVideoPipeline as VideoPipeline
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.utils.utils import load_state_dict

import torch
from diffusers.utils import export_to_video
from typing import List, Dict
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import os
import pickle
from teleai_pipe.utils.vision import (
    GeneratePoseControlImages,
    create_4grid_video,
)
import imageio
from PIL import Image
import numpy as np
import torch.nn as nn
import pickle


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
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504015_lora/models/checkpoint_epoch_2_step_2900/transformer/transformer_lora.pt"
lora_state_dict = load_state_dict(lora_model_path)
model_manager.load_lora("", lora_state_dict, lora_alpha=1.0)
pipe = VideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
pipe.dit.patch_embedding = nn.Conv3d(
    in_channels=52,
    out_channels=5120,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
    dtype=torch.bfloat16,
)
pipe.dit.load_state_dict(lora_state_dict, strict=False)
pipe.enable_vram_management(4 * 10**9)


def inference_pose_to_video(
    poses, input_image, prompt, seed, height, width, num_frames
):
    continue_num_frames = num_frames * 2  # 4段
    num_inference_steps = 40
    data_dict = {}
    data_dict["video_height"] = input_image.size[1]
    data_dict["video_width"] = input_image.size[0]
    data_dict["joints2d_nonparam"] = poses
    data_dict["sample_indexes"] = [
        0 + i * 2 for i in range(continue_num_frames) if 0 + i * 2 < len(poses)
    ]
    pose_generator = GeneratePoseControlImages()
    pose_images = pose_generator.call_nobox_noscore(data_dict)

    pose_images = [
        F.resize(img, (height, width), InterpolationMode.BILINEAR)
        for img in pose_images
    ]
    input_image = F.resize(input_image, (height, width), InterpolationMode.BILINEAR)

    final_videos = []
    video = []
    for start_idx in range(0, continue_num_frames, num_frames - 1):
        if len(pose_images[start_idx : start_idx + num_frames]) < num_frames:
            break
        input_image = input_image if start_idx == 0 else video[-1]
        video = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=5.0,
            seed=seed,
            tiled=True,
            tile_size=(34, 34),
            tile_stride=(18, 16),
            input_image=input_image,
            cn_images=pose_images[start_idx : start_idx + num_frames],
            model_type="Wan2.1-I2V-14B-720P",
            tea_cache_l1_thresh=0.3,
            attn_impl="sage_attn_8_8",
        )
        final_videos.extend(video[:-1])
    return final_videos, pose_images


pkl_paths = [
    (
        "/nvfile-heatstorage/hzh/Ps_motion/867_combined.pkl",
        "/nvfile-heatstorage/yfq/share/tmp/20250520_ps_demo/867_combined_ps.png",
        "Begin with a close-up of a man taking a bow on stage, dressed in a semi-formal outfit with a dark blazer, white shirt, and dark blue jeans. His expression is one of joy and accomplishment. The stage is lit with warm, ambient lighting, and the background features elegant, dark curtains. Transition the scene to show the audience in silhouette, applauding warmly. The camera then pans back to focus on the man as he straightens up, acknowledging the applause with a smile and a wave. Cut to a montage of moments from the performance that led to this bow, showcasing snippets of dialogue, laughter, and dramatic scenes.",
    ),
    (
        "/nvfile-heatstorage/hzh/Ps_motion/1028_combined.pkl",
        "/nvfile-heatstorage/yfq/share/tmp/20250520_ps_demo/1028_combined_ps.png",
        "Man in blazer and jeans performs a yoga pose on a mat. He's in a spacious room with white walls and natural light. His body forms a dynamic angle, balancing on one hand and knee, other limbs extended. He appears focused and composed.",
    ),
    (
        "/nvfile-heatstorage/hzh/Ps_motion/sport_tiaoshui_clip_0207_820_gt.pkl",
        "/nvfile-heatstorage/yfq/share/tmp/20250520_ps_demo/sport_tiaoshui_clip_0207_820_gt15_ps.png",
        "Man in a suit strides up stairs by the waterfront, then leaps off, causing a dramatic splash in the water below.",
    ),
    (
        "/nvfile-heatstorage/hzh/Ps_motion/sport_ticao_clip_0103_13_gt.pkl",
        "/nvfile-heatstorage/yfq/share/tmp/20250520_ps_demo/sport_ticao_clip_0103_13_gt_ps.png",
        "Man in blazer and jeans stands on a gymnastics mat by the pool, poised for performance, with officials and animated mascots in the background.",
    ),
    (
        "/nvfile-heatstorage/hzh/Ps_motion/wushu3.pkl",
        "/nvfile-heatstorage/yfq/share/tmp/20250520_ps_demo/wushu3_ps.png",
        "Man in a blazer and jeans performs a martial arts stance in front of a traditional Chinese gate, smiling broadly, with arms extended in dynamic pose.",
    ),
]

width = 1280
height = 720
num_frames = 45
output_dir = "work_dirs/output_demo0520_continue/"
os.makedirs(output_dir, exist_ok=True)


for idx, pkl_path in enumerate(pkl_paths):
    with open(pkl_path[0], "rb") as file:
        pose_data = pickle.load(file)["poses"]
    ref_image = Image.open(pkl_path[1]).convert("RGB")
    print(len(pose_data))
    print(pose_data[0].shape)
    video, pose_images = inference_pose_to_video(
        pose_data,
        ref_image,
        pkl_path[2],
        seed=0,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    export_to_video(video, output_dir + f"{idx}_cfg.mp4", fps=15)
    # 将pose,box,ref和生成的视频合并为一个视频进行展示
    combined_frames = create_4grid_video([pose_images, video])
    imageio.mimsave(
        output_dir + f"{idx}_combined.mp4",
        [np.array(frame) for frame in combined_frames],
        fps=16,
    )
