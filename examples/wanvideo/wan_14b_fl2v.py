import torch
import os
from PIL import Image
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.utils import get_example_video_name

PPL_CONFIG = dict(
    name="wan21_14B_fl2v",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.3,
    num_inference_steps=10,
    num_frames=81,
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
    seed=100,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
)


def get_pipeline():
    # Load models
    model_root = PPL_CONFIG["model_root"]
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,  # Image Encoder is loaded with float32
    )
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
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{model_root}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
    )
    pipe.enable_cpu_offload()
    return pipe


def run(pipeline, first_image, last_image, prompt, height, width):
    # First and last frame to video
    video = pipeline(
        prompt=prompt,
        input_image=first_image.resize((width, height)),
        end_image=last_image.resize((width, height)),
        negative_prompt=PPL_CONFIG["negative_prompt"],
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=PPL_CONFIG["seed"],
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
    )
    return video


if __name__ == "__main__":
    # Download example image
    prompt = "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door."
    first_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_F.jpg"
    )
    last_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_L.jpg"
    )
    pipe = get_pipeline()
    video = run(pipe, first_image, last_image, prompt, 720, 1080)
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
