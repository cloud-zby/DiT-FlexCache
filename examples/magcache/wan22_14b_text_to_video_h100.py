import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan22_video import WanVideo22Pipeline
from teleai_pipe.utils.video import save_video
import os
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.plugin.magcache.magcache import MagCacheCalibrator
import time

PPL_CONFIG = dict(
    name="wan22_A14B_i2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-A14B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    enable_magcache=False,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale_high=4,
    cfg_scale_low=3,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    sigma_shift=12.0,
    boundary=0.875,
    sample_solver="unipc",
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
        torch_dtype=torch.float32,
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
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=parallelism > 1,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.dit_high.cond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan2214b_t2v/shift12_cond.json",
    )
    pipe.dit_high.uncond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan2214b_t2v/shift12_uncond.json",
    )
    pipe.dit_low.cond_mag_cache = pipe.dit_high.cond_mag_cache
    pipe.dit_low.uncond_mag_cache = pipe.dit_high.uncond_mag_cache
    pipe.enable_cpu_offload()
    return pipe


def run(pipeline, prompt, height, width, negative_prompt="", seed=PPL_CONFIG["seed"]):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        image (PIL.Image/ndarray): 输入图像，分辨率需匹配height/width参数
        prompt (str): 正向引导文本提示词
        height (int): 输出视频高度(像素)
        width (int): 输出视频宽度(像素)
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    video = pipeline(
        prompt=prompt,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
        boundary=PPL_CONFIG["boundary"],
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(1)
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    video = run(pipe, prompt, 720, 1280)
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
