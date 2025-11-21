import torch
import os
from PIL import Image
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline, get_target_image_size
from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
from teleai_pipe.plugin.magcache.magcache import MagCacheCalibrator

PPL_CONFIG = dict(
    name="wan21_14B_fl2v",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=5.0,
    resolution="720p",
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    sigma_shift=16.0,
    prompt_expander_config=dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="zh",
    ),
    sample_solver="euler",
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
)


def get_pipeline(parallelism=1, with_prompt_expander=False):
    """
    Args:
        parallelism (int): 推理并行卡数目 1, 2, 4 or 8
    """
    assert parallelism in [1, 2, 4, 8]
    # Load models
    model_root = PPL_CONFIG["model_root"]
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )
    model_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.float32,  # VAE is loaded with float32
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
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=False,
        sample_solver=PPL_CONFIG["sample_solver"],
        vae_dtype=torch.float32,
    )
    pipe.dit.cond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan14b_fl2v/shift16_cond.json",
    )
    pipe.dit.uncond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan14b_fl2v/shift16_uncond.json",
    )
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
    first_image,
    last_image,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
):
    """
        使用视频生成模型将首尾图像转换为视频序列, 首尾帧图像的比例需要保持一致
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        first_image (PIL.Image/ndarray): 首帧图像
        last_image (PIL.Image/ndarray):末帧图像
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        resolution(str): 分辨率 such as 720p, 480p

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    width, height = get_target_image_size(
        first_image.size[0], first_image.size[1], resolution=resolution
    )
    if isinstance(pipeline, tuple):
        real_pipe, prompt_expander = pipeline
        origin_prompt = prompt
        prompt = prompt_expander(
            prompt,
            image=[first_image, last_image],
            seed=100,
            tar_lang=PPL_CONFIG["prompt_expander_config"]["target_lan"],
        ).prompt
        print(f"extend {origin_prompt} to  {prompt}")
    else:
        real_pipe = pipeline
    # First and last frame to video
    video = real_pipe(
        prompt=prompt,
        input_image=first_image.resize((width, height)),
        end_image=last_image.resize((width, height)),
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
    )
    return video


if __name__ == "__main__":
    prompt = "航拍视角下，一座现代体育馆从初始的钢筋玻璃结构，逐渐被自然植物覆盖，最终变成被藤蔓、花朵和树木缠绕的“生态穹顶”，背景光线从白天过渡到黄昏，整体充满生命力与魔幻感。"
    first_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_F_building.png"
    ).convert("RGB")
    last_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_L_building.png"
    ).convert("RGB")
    pipe = get_pipeline(1, False)
    video = run(pipe, first_image, last_image, prompt)
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=6,
    )
    del pipe
