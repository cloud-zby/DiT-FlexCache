import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
import os
import torch.multiprocessing as mp
from teleai_pipe.utils.utils import get_example_video_name
import time
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
from teleai_pipe.pipelines.wan_video import get_target_video_size_from_ratio
from loguru import logger

PPL_CONFIG = dict(
    name="wan21_14B_t2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    num_inference_steps=50,
    resolution="720p",
    num_frames=81,
    cfg_scale=5.0,
    enable_magcache=True,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
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
    prompt_expander_config=dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="zh",
    ),
    sample_solver="unipc",
)


def get_pipeline(parallelism=1, with_prompt_expander=False):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
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
        vae_parallel=True,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.enable_cpu_offload()
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
    pipe = get_pipeline(2, with_prompt_expander=True)
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    aspect_ratio = "16:9"
    video = run(pipe, prompt, aspect_ratio=aspect_ratio)
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
