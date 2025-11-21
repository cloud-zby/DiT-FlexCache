import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_ref_v1 import (
    TeleAIVideoRefV1,
    get_target_video_size_from_ratio,
)
from teleai_pipe.pipelines.teleai_video_sr import TeleVideoSRPipeline
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
import time

PPL_CONFIG = dict(
    name="teleai_ref_v1",
    model_root="/nvfile-heatstorage/model_zoo/release/video/teleai_video_r2v_251030",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    base_num_inference_steps=40,
    sr_num_inference_steps=10,
    num_frames=81,
    resolution="720p",
    aspect_ratio="16:9",
    base_cfg_scale=7.5,  # text
    base_cfg_scale_img=5.0,
    sr_cfg_scale=5.5,
    base_timestep_range=(333, 1000),
    sr_timestep_range=(0, 500),
    base_sigma_shift=5,
    sr_sigma_shift=5,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="TeleAI-S2V-14B",
    sample_solver="euler",  # unipc
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_r2v_251030/dit_model_bfloat16_2f2f4.safetensors"
    ],
    sr_dit_path_list=[
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_2k_25080922/diffusion_pytorch_model_d5916c.bin"
    ],
    enable_dit_magcache=False,
    enable_sr_magcache=False,
)


def get_pipeline(parallelism=1):
    """
    Args:
        parallelism (int): 推理并行卡数目 3
    """
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
    pipe = TeleAIVideoRefV1.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 3),
        use_cfg_parallel=(parallelism >= 3),
        vae_parallel=False,
        vae_dtype=torch.bfloat16,
        sample_solver=PPL_CONFIG["sample_solver"],
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
    )
    sr_pipe.enable_cpu_offload()
    pipe.sr_pipe = sr_pipe
    return pipe


def run(
    pipeline,
    ref_images,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio=PPL_CONFIG["aspect_ratio"],
):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        image (PIL.Image/ndarray): 输入图像，分辨率需匹配height/width参数
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        resolution(str): 分辨率 such as 720p, 480p

    Returns:
        List[PIL.Image]: 生成的视频序列
    """

    width, height = get_target_video_size_from_ratio(
        aspect_ratio=aspect_ratio, resolution=resolution
    )
    video = pipeline(
        prompt=prompt,
        ref_images=ref_images,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        base_num_inference_steps=PPL_CONFIG["base_num_inference_steps"],
        sr_num_inference_steps=PPL_CONFIG["sr_num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        base_cfg_scale=PPL_CONFIG["base_cfg_scale"],
        base_cfg_scale_img=PPL_CONFIG["base_cfg_scale_img"],
        sr_cfg_scale=PPL_CONFIG["sr_cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        base_sigma_shift=PPL_CONFIG["base_sigma_shift"],
        sr_sigma_shift=PPL_CONFIG["sr_sigma_shift"],
        base_timestep_range=PPL_CONFIG["base_timestep_range"],
        sr_timestep_range=PPL_CONFIG["sr_timestep_range"],
        sr_scale=1 if resolution == "480p" else 2,
        enable_sr_magcache=PPL_CONFIG["enable_sr_magcache"],
        enable_dit_magcache=PPL_CONFIG["enable_dit_magcache"],
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(3)
    image_paths = [
        f"{os.path.dirname(__file__)}/../data/ref12.png",
        f"{os.path.dirname(__file__)}/../data/ref13.png",
    ]
    ref_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    prompt = "扎着双丸子头，身着红黑配色并带有火焰纹饰服饰，颈戴金项圈、臂缠金护腕的哪吒，和有着一头淡蓝色头发，额间有蓝色印记，身着一袭白色长袍的敖丙，并肩坐在教室的座位上，他们专注地讨论着书本内容。背景为柔和的灯光和窗外微风拂过的树叶，营造出安静又充满活力的学习氛围。"

    start = time.time()
    video = run(pipe, ref_images, prompt)
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
