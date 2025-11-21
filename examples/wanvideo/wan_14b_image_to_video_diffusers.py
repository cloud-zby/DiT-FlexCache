import torch
import os
from PIL import Image
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video

model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16,  # Image Encoder is loaded with float16
)
dit_model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-I2V-14B-720P-Diffusers/transformer"
dit_model_path_list = os.listdir(dit_model_root)
dit_model_path_list = [
    os.path.join(dit_model_root, item)
    for item in dit_model_path_list
    if item.endswith(".safetensors")
]
model_manager.load_models(
    [
        dit_model_path_list,
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_root}/Wan2.1_VAE.pth",
    ],
    # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=False
)
pipe.enable_vram_management(
    num_persistent_param_in_dit=None
)  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

import os

image = Image.open(f"{os.path.dirname(__file__)}/../data/wan_input.jpg")

# Image-to-video
prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    tea_cache_l1_thresh=0.15,
    num_inference_steps=10,
    num_frames=49,
    cfg_scale=1.0,
    seed=0,
    tiled=True,
    height=720,
    width=1280,
    attn_impl="cudnn",
    model_type="Wan2.1-I2V-14B-720P",
)
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    tea_cache_l1_thresh=0.15,
    num_inference_steps=50,
    num_frames=49,
    cfg_scale=1.0,
    seed=0,
    tiled=True,
    height=720,
    width=1280,
    attn_impl="cudnn",
    model_type="Wan2.1-I2V-14B-720P",
)
save_video(video, "video_oven.mp4", fps=16, quality=6)
