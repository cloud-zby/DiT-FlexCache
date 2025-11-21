from teleai_pipe.pipelines.teleai_video_v1.pipeline_diffusers import VideoPipeline
import torch
from diffusers.utils import export_to_video
from typing import List, Dict
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import os
from teleai_pipe.plugin.enhance_a_video import (
    enable_enhance,
    inject_enhance_for_hunyuanvideo,
    set_enhance_weight,
)


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


ref_frames = {0: f"{os.path.dirname(__file__)}/../data/wan_input.jpg"}
# ref_frames = {0: f"{os.path.dirname(__file__)}/../data/sand_and_girl.png"}
base_model_path = "/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
transformer_model_path = "/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_720p_33_45_dp160_nogym/models/checkpoint_epoch_1_step_5000/transformer_safetensor"
(
    width,
    height,
) = 1280, 720
prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
# prompt = "a girl in a cloak is seen from behind, squatting at the edge of a rocky cave in the midst of a desert. Her gaze is directed towards the vast and boundless expanse of the desert. In the distance, layers upon layers of sandstone mountains with various shapes can be seen. A few wisps of white clouds drift across the sky, creating an overall atmosphere of solitude and tranquility."
device = torch.device("cuda:0")
print(f"prompt: {prompt}")


pipeline = VideoPipeline.from_pretrained(
    base_model_path,
    transformer_model_path=transformer_model_path,
    torch_dtype=torch.bfloat16,
    disable_mmap=True,
).to(device)
pipeline.vae.enable_tiling()
# comment the following if you want to use the original model
# inject_enhance_for_hunyuanvideo(pipeline.transformer)
# enhance_weight can be adjusted for better visual quality
# set_enhance_weight(0)
# enable_enhance()

num_frames = 49
num_inference_steps = 50
attn_impl = "cudnn"
ref_images = prepare_reference_images(width, height, num_frames, ref_frames)
# warm up, 这里只是用于测试， 正式使用时不需要加这个
video = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=10,
    seed=42,
    model_type="i2vhy",
    ref_images=ref_images,
    guidance_scale=6.0,
    teacache_l1_th=0.15,
    attn_impl=attn_impl,
).frames[0]
# 这里是正式测试
video = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    seed=42,
    model_type="i2vhy",
    ref_images=ref_images,
    guidance_scale=6.0,
    teacache_l1_th=0.15,
    attn_impl=attn_impl,
).frames[0]

export_to_video(video, "diffusersv1_cfg6.mp4", fps=15)
