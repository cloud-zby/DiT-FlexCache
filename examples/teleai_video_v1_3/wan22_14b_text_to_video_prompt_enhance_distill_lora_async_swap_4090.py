import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan22_video import WanVideo22Pipeline
from teleai_pipe.pipelines.wan_video import get_target_video_size_from_ratio
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
import time

PPL_CONFIG = dict(
    name="wan22_A14B_t2v_distill_lora_4090",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    enable_magcache_dit_high=True,
    enable_magcache_dit_low=False,
    num_inference_steps=(40, 4),
    num_frames=81,
    resolution="720p",
    cfg_scale_high=4.0,
    cfg_scale_low=1.0,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8",
    model_type="Wan2.2-T2V-A14B",
    sigma_shift=(12.0, 5.0),
    boundary=0.875,
    sample_solver="multi_euler",
    dit_high_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    dit_low_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    prompt_expander_config=dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="zh",
    ),
    lora_configs={
        "dit_low": [
            {
                "path": "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
                "strength": 1.0,
            }
        ],
    },
    enable_dit_async_swap=True,
    async_swap_config={
        "offload_granularity": "block",  # "block" or "phase"
        "clean_cuda_cache": False,
    },
)


def get_pipeline(parallelism=1, with_prompt_expander=False):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    # vae
    model_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    # dit high
    model_manager.load_model(
        PPL_CONFIG["dit_high_path_list"],
        torch_dtype=torch.bfloat16,
    )
    # dit low
    model_manager.load_model(
        PPL_CONFIG["dit_low_path_list"],
        torch_dtype=torch.bfloat16,
    )

    # t5
    model_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideo22Pipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        parallelism=parallelism,
        dit_high_compile=False,
        dit_low_compile=False,
        dit_high_fsdp=False,
        dit_low_fsdp=False,
        dit_high_cfg_parallel=(parallelism >= 2),
        dit_low_cfg_parallel=False,
        vae_parallel=True,
        vae_dtype=torch.bfloat16,
        sample_solver=PPL_CONFIG["sample_solver"],
        lora_configs=PPL_CONFIG["lora_configs"],
        enable_dit_async_swap=PPL_CONFIG["enable_dit_async_swap"],
        dit_async_swap_config=PPL_CONFIG["async_swap_config"],
    )
    pipe.enable_vram_management()
    if with_prompt_expander:
        prompt_config = PPL_CONFIG["prompt_expander_config"]
        prompt_expander = OpenAIPromptExpander(
            model_name=prompt_config["name"], address=prompt_config["url"], is_vl=False
        )
        return pipe, prompt_expander
    return pipe


def run(
    pipeline,
    prompt,
    aspect_ratio="16:9",
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        aspect_ratio (str): width:height, such as 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 1:1
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        resolution(str): 分辨率 such as 720p, 480p

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio, resolution=resolution
    )
    if isinstance(pipeline, tuple):
        real_pipe, prompt_expander = pipeline
        origin_prompt = prompt
        prompt = prompt_expander(
            prompt,
            seed=100,
            tar_lang=PPL_CONFIG["prompt_expander_config"]["target_lan"],
        ).prompt
        print(f"extend {origin_prompt} to  {prompt}")
    else:
        real_pipe = pipeline
    video = real_pipe(
        prompt=prompt,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache_dit_high=PPL_CONFIG["enable_magcache_dit_high"],
        enable_magcache_dit_low=PPL_CONFIG["enable_magcache_dit_low"],
        boundary=PPL_CONFIG["boundary"],
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(8, True)
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
