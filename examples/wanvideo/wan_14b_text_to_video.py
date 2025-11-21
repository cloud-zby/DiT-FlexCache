import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.utils import get_example_video_name
import os
import time

PPL_CONFIG = dict(
    name="wan21_14B_t2v",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.1,
    num_inference_steps=50,
    num_frames=81,
    cfg_scale=5.0,
    cfg_skip_ratio=0.1,
    seed=100,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-T2V-14B",
    enable_magcache=False,
)


# Download models
# snapshot_download("Wan-AI/Wan2.1-T2V-14B", cache_dir="models")
def get_pipeline():
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                f"{model_root}/diffusion_pytorch_model-00001-of-00006.safetensors",
                f"{model_root}/diffusion_pytorch_model-00002-of-00006.safetensors",
                f"{model_root}/diffusion_pytorch_model-00003-of-00006.safetensors",
                f"{model_root}/diffusion_pytorch_model-00004-of-00006.safetensors",
                f"{model_root}/diffusion_pytorch_model-00005-of-00006.safetensors",
                f"{model_root}/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{model_root}/Wan2.1_VAE.pth",
        ],
        # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_cpu_offload()
    pipe.enable_compile_dit()
    return pipe


# Text-to-video
def run(pipeline, prompt, height, width):
    start = time.time()
    video = pipeline(
        prompt=prompt,
        negative_prompt=PPL_CONFIG["negative_prompt"],
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=PPL_CONFIG["seed"],
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
    )
    print(f"generate video cost {time.time()-start}s")
    return video


if __name__ == "__main__":
    prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
    prompt = "On the ship, Jack and Rose were dancing on the deck, surrounded by a jubilant crowd."
    prompt = "烽火台上，士兵们紧张而有序地操作着，点燃烽火，火光在远处的山峦间跳跃，映照出他们的紧张表情。"
    pipe = get_pipeline()
    # warmup
    video = pipe(
        prompt=prompt,
        negative_prompt=PPL_CONFIG["negative_prompt"],
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=10,
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=PPL_CONFIG["seed"],
        tiled=PPL_CONFIG["tiled"],
        height=720,
        width=1280,
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
    )
    video = run(pipe, prompt, 720, 1280)
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
