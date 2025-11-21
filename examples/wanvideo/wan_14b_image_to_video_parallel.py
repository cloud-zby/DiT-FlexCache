import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
import os
import torch.multiprocessing as mp
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name


# Load models
# model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-T2V-1.3B"
if __name__ == "__main__":
    model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
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
        # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=True,
        parallelism=4,
        compile=False,
        use_cfg_parallel=False,
        vae_parallel=False,
    )
    pipe.enable_cpu_offload()
    # pipe.enable_vram_management(0)
    import time

    start = time.time()
    prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
    image = Image.open(f"{os.path.dirname(__file__)}/../data/wan_input.jpg")
    video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=40,
        cfg_scale=5.0,
        input_image=image,
        seed=0,
        tiled=True,
        num_frames=81,
        height=720,
        width=1280,
        attn_impl="sage_attn_8_8_sm90",  # "sage_attn_8_8" on 4090
        tea_cache_l1_thresh=0.0,  # The larger this value is, the faster the speed, but the worse the visual quality.
        enable_magcache=True,
        model_type="Wan2.1-I2V-14B-720P",
    )
    print(f"generate video cost {time.time()-start}s")
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
    print("finish save video")
    del pipe
