import torch
from PIL import Image
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_vace_video import WanVaceVideoPipeline
from teleai_pipe.utils.video import save_video, VideoData
from teleai_pipe.utils.utils import get_example_video_name

model_root = "/workspace/Wan2.1-VACE-14B"

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        [
            f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00003-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00004-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00005-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00006-of-00007.safetensors",
            f"{model_root}/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        # "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-VACE-1.3B-Preview/diffusion_pytorch_model.safetensors",
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_root}/Wan2.1_VAE.pth",
    ],
    # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
pipe = WanVaceVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
)
pipe.enable_cpu_offload()

import os

image = Image.open("/nvfile-heatstorage/zuoxin/wan_example/cat_fightning.jpg").resize(
    (1280, 720)
)
control_video = VideoData("/nvfile-heatstorage/zuoxin/wan_example/depth_video.mp4")
# Image-to-video
prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.3,
    num_inference_steps=50,
    vace_reference_image=image,
    vace_video=control_video,
    num_frames=81,
    cfg_scale=5.0,
    seed=1,
    tiled=True,
    height=720,
    width=1280,
    attn_impl="cudnn",
    model_type="Wan2.1-I2V-14B-720P",
)
save_video(
    video,
    os.path.join(
        os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"), get_example_video_name(__file__)
    ),
    fps=16,
    quality=6,
)
