import os
import torch
from PIL import Image
import os
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1_4.pipeline import WanVideoPipeline
from teleai_pipe.utils.video import save_video


def get_pipeline():
    model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )

    model_manager.load_models(
        [
            "/nvfile-heatstorage/model_zoo/mmgen/camcontrol_wan/diffusion_pytorch_model.safetensors",
            # "/home/Wan2.1-Fun-V1.1-14B-Control-Camera/diffusion_pytorch_model.safetensors",
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{model_root}/Wan2.1_VAE.pth",
        ],
        # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
    )
    pipe.enable_cpu_offload()

    return pipe


def inferenece_one_video(
    path_save, pipeline, prompt, camera_movement_type, start_image=None
):
    video = pipeline(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=start_image,
        tea_cache_l1_thresh=0.3,
        num_inference_steps=50,
        num_frames=49,
        cfg_scale=5.0,
        seed=99,
        tiled=True,
        height=720,
        width=1280,
        attn_impl="cudnn",
        model_type="Wan2.1-I2V-14B-720P",
        camera_movement_type=camera_movement_type,
    )

    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    save_video(video, path_save, fps=16)
    print(f"Saved to {path_save}")
