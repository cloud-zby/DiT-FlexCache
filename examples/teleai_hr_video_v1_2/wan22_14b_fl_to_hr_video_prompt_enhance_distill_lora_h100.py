import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan22_video import WanVideo22Pipeline
from teleai_pipe.utils.video import save_video
import os
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.pipelines.wan_video import get_target_image_size
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
from teleai_pipe.pipelines.teleai_video_sr import TeleVideoSRPipeline
import time
from loguru import logger
from PIL import Image

PPL_CONFIG = dict(
    name="wan22_A14B_fl2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    enable_magcache_dit_high=True,
    enable_magcache_dit_low=False,
    num_inference_steps=(40, 4),
    num_frames=81,
    resolution="1080p",
    cfg_scale_high=3.5,
    cfg_scale_low=1.0,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.2-FL2V-A14B",
    sigma_shift=(5.0, 1.0),
    boundary=0.9,
    sample_solver="multi_euler",
    dit_high_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP/high_noise_model/diffusion_pytorch_model.safetensors",
    ],
    dit_low_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP/low_noise_model/diffusion_pytorch_model.safetensors",
    ],
    sr_dit_path_list=[
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_2k_25080922/diffusion_pytorch_model_d5916c.bin"
    ],
    lora_configs={
        "dit_low": [
            {
                "path": "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
                "strength": 1.0,
            }
        ],
    },
)


def get_pipeline(parallelism=1):
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
        dit_low_fsdp=False,
        dit_high_fsdp=False,
        parallelism=parallelism,
        dit_high_compile=True,
        dit_low_compile=False,
        dit_high_cfg_parallel=(parallelism >= 2),
        dit_low_cfg_parallel=False,
        vae_parallel=(parallelism >= 2),
        vae_dtype=torch.bfloat16,
        sample_solver=PPL_CONFIG["sample_solver"],
        lora_configs=PPL_CONFIG["lora_configs"],
    )
    pipe.enable_cpu_offload()
    sr_model_manager = ModelManager(device="cpu")
    sr_model_manager.load_models(
        [PPL_CONFIG["sr_dit_path_list"]],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    sr_pipe = TeleVideoSRPipeline.from_model_manager(
        model_manager=sr_model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=False,
        vae_dtype=torch.float32,
    )
    sr_pipe.enable_cpu_offload()
    pipe.sr_pipe = sr_pipe
    return pipe


def run(
    pipeline,
    first_image,
    last_image,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        first_image (PIL.Image/ndarray): 首帧图像
        last_image (PIL.Image/ndarray):末帧图像
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        resolution(str): 分辨率 such as 720p, 480p, 1080p, 2k

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    width, height = get_target_image_size(
        first_image.size[0], first_image.size[1], resolution=resolution
    )
    video = pipeline(
        prompt=prompt,
        input_image=first_image,
        end_image=last_image,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        seed=seed,
        tiled=True if resolution == "2k" else False,
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache_dit_high=PPL_CONFIG["enable_magcache_dit_high"],
        enable_magcache_dit_low=PPL_CONFIG["enable_magcache_dit_low"],
        boundary=PPL_CONFIG["boundary"],
        sr_scale=1 if resolution in ["480p", "720p"] else 2,
        tile_size=(60, 104),
        tile_stride=(30, 52),
    )
    return video


def run_with_file(
    pipeline,
    first_image,
    last_image,
    prompt,
    negative_prompt,
    seed,
    target_video_resolution,
    save_video_path,
    enable_prompt_enhance,
    **kwargs,
):
    video = run(
        pipeline,
        first_image,
        last_image,
        prompt,
        negative_prompt,
        seed,
        target_video_resolution,
        enable_prompt_enhance,
    )
    logger.info(f"save target video to {save_video_path}")
    save_video(
        video,
        save_video_path,
        fps=16,
        quality=6,
    )


if __name__ == "__main__":
    prompt_config = dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="zh",
        movie_style=True,
    )
    prompt_expander = OpenAIPromptExpander(
        model_name=prompt_config["name"], address=prompt_config["url"], is_vl=True
    )
    pipe = get_pipeline(2)
    prompt = "航拍视角下，一座现代体育馆从初始的钢筋玻璃结构，逐渐被自然植物覆盖，最终变成被藤蔓、花朵和树木缠绕的“生态穹顶”，背景光线从白天过渡到黄昏，整体充满生命力与魔幻感。"
    first_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_F_building.png"
    ).convert("RGB")
    last_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_L_building.png"
    ).convert("RGB")
    origin_prompt = prompt
    prompt = prompt_expander(
        prompt,
        image=[first_image, last_image],
        seed=100,
        tar_lang=prompt_config["target_lan"],
    ).prompt
    print(f"extend {origin_prompt} to  {prompt}")
    start = time.time()
    video = run(pipe, first_image, last_image, prompt)
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
