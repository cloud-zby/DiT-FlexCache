import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_ref_v1 import (
    TeleAIVideoRefV1,
    get_target_image_size,
    load_ref_images_and_size,
)
from teleai_pipe.pipelines.teleai_video_sr import TeleVideoSRPipeline
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
import time
from teleai_pipe.plugin.magcache.magcache import MagCacheCalibrator


PPL_CONFIG = dict(
    name="teleai_ref_v1",
    model_root="/root/Wan2.1-T2V-14B",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    base_num_inference_steps=40,
    sr_num_inference_steps=10,
    num_frames=81,
    resolution="480p",
    aspect_ratio="16:9",
    base_cfg_scale=7.5, # text
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
    sample_solver="euler", # unipc
    dit_path_list=[
        "/root/Phatom14B/Phantom_Wan_14B-00001-of-00006.safetensors",
        "/root/Phatom14B/Phantom_Wan_14B-00002-of-00006.safetensors",
        "/root/Phatom14B/Phantom_Wan_14B-00003-of-00006.safetensors",
        "/root/Phatom14B/Phantom_Wan_14B-00004-of-00006.safetensors",
        "/root/Phatom14B/Phantom_Wan_14B-00005-of-00006.safetensors",
        "/root/Phatom14B/Phantom_Wan_14B-00006-of-00006.safetensors",
    ],
    # sr_dit_path_list=[
    #     "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_2k_25080922/diffusion_pytorch_model_d5916c.bin"
    # ],
    enable_dit_magcache=True,
    enable_sr_magcache=False,
    prompt_expander_config=dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="en",
    ),
)


def get_pipeline(parallelism=1, with_prompt_expander=False):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    # model_manager.load_models(
    #     [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    #     torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    # )
    model_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,  # VAE is loaded with float32
    )
    model_manager.load_models(
        [PPL_CONFIG["dit_path_list"]],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    # model_manager.load_models(
    #     [PPL_CONFIG["sr_dit_path_list"]],
    #     torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    # )
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
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=False,
        vae_dtype=torch.bfloat16,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.dit.cond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["base_num_inference_steps"],
        "work_dirs/magecache/teleai14_s2v/shift5_cond.json",
    )
    pipe.dit.uncond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["base_num_inference_steps"],
        "work_dirs/magecache/teleai14_s2v/shift5_uncond.json",
    )
    # pipe.sr_dit.cond_mag_cache = MagCacheCalibrator(
    #     PPL_CONFIG["sr_num_inference_steps"],
    #     "work_dirs/magecache/teleai_sr/shift5_cond.json",
    # )
    # pipe.sr_dit.uncond_mag_cache = MagCacheCalibrator(
    #     PPL_CONFIG["sr_num_inference_steps"],
    #     "work_dirs/magecache/teleai_sr/shift5_uncond.json",
    # )
    pipe.enable_cpu_offload()
    if with_prompt_expander:
        prompt_config = PPL_CONFIG["prompt_expander_config"]
        prompt_expander = OpenAIPromptExpander(
            model_name=prompt_config["name"], address=prompt_config["url"], is_vl=True
        )
        return pipe, prompt_expander
    return pipe


def run(
    pipeline,
    ref_images,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio=PPL_CONFIG["aspect_ratio"],
    enable_prompt_enhance: bool = True,
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
        enable_prompt_enhance(bool): 是否使用prompt enhance

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    ref_images, width, height = load_ref_images_and_size(ref_images, resolution=resolution, aspect_ratio=aspect_ratio)

    if isinstance(pipeline, tuple):
        real_pipe, prompt_expander = pipeline
        if enable_prompt_enhance:
            origin_prompt = prompt
            prompt = prompt_expander(
                prompt,
                # image=image,
                seed=100,
                tar_lang=PPL_CONFIG["prompt_expander_config"]["target_lan"],
            ).prompt
            print(f"extend {origin_prompt} to  {prompt}")
    else:
        real_pipe = pipeline
    video = real_pipe(
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
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(1, False)
    image_paths = [
        f"{os.path.dirname(__file__)}/../data/ref12.png",
        f"{os.path.dirname(__file__)}/../data/ref13.png"
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
