import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
from teleai_pipe.pipelines.wan_video import get_target_video_size_from_ratio
import os
from teleai_pipe.utils.utils import get_example_video_name
from loguru import logger
import time

PPL_CONFIG = dict(
    name="wan21_14B_t2v_4090",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    num_inference_steps=50,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
    enable_magcache=True,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8",
    model_type="Wan2.1-T2V-14B",
    sigma_shift=5.0,
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    sample_solver="unipc",
)


def get_pipeline(parallelism=8):
    """
    Args:
        parallelism (int): 推理并行卡数目 4 or 8
    """
    assert parallelism in [4, 8]
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,  # VAE is loaded with bfloat16
    )
    model_manager.load_models(
        [PPL_CONFIG["dit_path_list"]],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    model_manager.load_models(
        [
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=True,
        parallelism=parallelism,
        compile=True,
        use_cfg_parallel=(parallelism == 8),
        vae_parallel=True,
        vae_dtype=torch.bfloat16,
        vae_device_ids=list(range(1, parallelism)),
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.enable_vram_management()
    return pipe


def run(
    pipeline,
    prompt,
    aspect_ratio: str = "16:9",
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        prompt (str): 正向引导文本提示词
        aspect_ratio (str): 输出视频的比例，具体如下：
            "16:9"
            "9:16"
            "4:3"
            "3:4"
            "1:1"
            "2:3"
            "3:2"
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        reslution (str): 720p, 480p

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio, resolution=resolution
    )
    video = pipeline(
        prompt=prompt,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(8)
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    video = run(pipe, prompt)
    print(f"generate video cost {time.time()-start}s")
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=6,
    )
    print("finish save video")
    del pipe
