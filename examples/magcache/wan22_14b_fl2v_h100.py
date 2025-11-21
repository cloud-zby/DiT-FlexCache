import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan22_video import WanVideo22Pipeline
from teleai_pipe.pipelines.wan_video import get_target_image_size
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
import time
from teleai_pipe.plugin.magcache.magcache import MagCacheCalibrator

PPL_CONFIG = dict(
    name="wan22_A14B_fl2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    enable_magcache=False,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale_high=3.5,
    cfg_scale_low=3.5,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="euler",
    dit_high_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP/high_noise_model/diffusion_pytorch_model.safetensors",
    ],
    dit_low_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Fun-A14B-InP/low_noise_model/diffusion_pytorch_model.safetensors",
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
        parallelism=parallelism,
        vae_parallel=False,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.dit_high.cond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan2214b_fl2v/shift5_cond_euler.json",
    )
    pipe.dit_high.uncond_mag_cache = MagCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        "work_dirs/magecache/wan2214b_fl2v/shift5_uncond_euler.json",
    )
    pipe.dit_low.cond_mag_cache = pipe.dit_high.cond_mag_cache
    pipe.dit_low.uncond_mag_cache = pipe.dit_high.uncond_mag_cache
    pipe.enable_cpu_offload()
    return pipe


def run(
    pipeline,
    first_image,
    last_image,
    prompt,
    height,
    width,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
):
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
        input_image=first_image,
        end_image=last_image,
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
        boundary=PPL_CONFIG["boundary"],
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(1)
    prompt = "航拍视角下，一座现代体育馆从初始的钢筋玻璃结构，逐渐被自然植物覆盖，最终变成被藤蔓、花朵和树木缠绕的“生态穹顶”，背景光线从白天过渡到黄昏，整体充满生命力与魔幻感。"
    first_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_F_building.png"
    ).convert("RGB")
    last_image = Image.open(
        f"{os.path.dirname(__file__)}/../data/FL_1_L_building.png"
    ).convert("RGB")
    width, height = get_target_image_size(first_image.size[0], first_image.size[1])
    start = time.time()
    video = run(pipe, first_image, last_image, prompt, height, width)
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
