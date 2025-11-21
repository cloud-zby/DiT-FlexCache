import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline, get_target_image_size
from teleai_pipe.utils.video import save_video
import os
import torch.multiprocessing as mp
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
import time
from teleai_pipe.plugin.teacache.teacache import TeaCacheCalibrator

PPL_CONFIG = dict(
    name="wan21_14B_i2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    enable_magcache=False,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    sigma_shift=3.0,
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    # "/nvfile-heatstorage/ljq/models/ckpts/telestudio_i2v_wan_step1000/diffusion_pytorch_model.pth"
)


def get_pipeline(parallelism=1, output_dir="work_dirs"):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
    model_root = PPL_CONFIG["model_root"]
    # Load models
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
        compile=False,
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=True,
        vae_dtype=torch.float32,
    )

    cond_cache_path = os.path.join(output_dir, "teacache/wan14b_i2v/cond.json")
    uncond_cache_path = os.path.join(output_dir, "teacache/wan14b_i2v/uncond.json")

    pipe.dit.cond_tea_cache = TeaCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        cond_cache_path,
        model_type=PPL_CONFIG["model_type"],
        prompt_id=0,
    )
    pipe.dit.uncond_tea_cache = TeaCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        uncond_cache_path,
        model_type=PPL_CONFIG["model_type"],
        prompt_id=0,
    )
    pipe.enable_cpu_offload()
    return pipe


def run(
    pipeline,
    image,
    prompt,
    height,
    width,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    prompt_id=0,
    output_dir="work_dirs",
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

    cond_cache_path = os.path.join(output_dir, "teacache/wan14b_i2v/cond.json")
    uncond_cache_path = os.path.join(output_dir, "teacache/wan14b_i2v/uncond.json")

    pipeline.dit.cond_tea_cache = TeaCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        cond_cache_path,
        model_type=PPL_CONFIG["model_type"],
        prompt_id=prompt_id,
    )
    pipeline.dit.uncond_tea_cache = TeaCacheCalibrator(
        PPL_CONFIG["num_inference_steps"],
        uncond_cache_path,
        model_type=PPL_CONFIG["model_type"],
        prompt_id=prompt_id,
    )
    video = pipeline(
        prompt=prompt,
        input_image=image,
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
    # 注意，以下两行必须在 __name__ == "__main__" 下面执行，保证执行一次
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    pipe = get_pipeline(1)
    image = Image.open(
        f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png"
    ).convert("RGB")
    # prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    # width, height = get_target_image_size(image.size[0], image.size[1])
    # start = time.time()
    # video = run(pipe, image, prompt, height, width)
    # print(f"generate video cost {time.time()-start}s")
    # save_video(
    #     video,
    #     os.path.join(
    #         os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
    #         get_example_video_name(__file__),
    #     ),
    #     fps=16,
    #     quality=6,
    # )
    # print("finish save video")
    # del pipe

    prompts = [
        "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
        "A cute little girl playing with her dog in the garden, with warm sunlight creating a peaceful atmosphere.",
        "The little girl and her dog running together happily in the backyard, expressing pure joy and friendship.",
        "A tender moment as the girl sits with her dog under the shade of a tree, enjoying a quiet afternoon.",
    ]
    width, height = get_target_image_size(image.size[0], image.size[1])

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")

    for i, prompt in enumerate(prompts):
        print(f"\n=== 正在处理第 {i+1}/{len(prompts)} 个prompt ===")
        print(f"Prompt: {prompt}")
        cond_dir = "work_dirs/teacache/wan14b_i2v/cond.json"
        uncond_dir = "work_dirs/teacache/wan14b_i2v/uncond.json"

        # 传入对应的prompt_id
        pipe.dit.cond_tea_cache = TeaCacheCalibrator(
            PPL_CONFIG["num_inference_steps"],
            cond_dir,
            model_type=PPL_CONFIG["model_type"],
            prompt_id=i,
        )
        pipe.dit.uncond_tea_cache = TeaCacheCalibrator(
            PPL_CONFIG["num_inference_steps"],
            uncond_dir,
            model_type=PPL_CONFIG["model_type"],
            prompt_id=i,
        )

        start_time = time.time()
        video = run(pipe, image, prompt, height, width, seed=PPL_CONFIG["seed"])
        generation_time = time.time() - start_time
        print(f"generate video cost {time.time()-start_time}s")

        video_filename = (
            f"video_{i+1:03d}_{get_example_video_name(__file__).split('.')[0]}.mp4"
        )
        video_path = os.path.join(output_dir, video_filename)
        save_video(video, video_path, fps=16, quality=6)
        print(f"finish save video: {video_path}")
    del pipe
