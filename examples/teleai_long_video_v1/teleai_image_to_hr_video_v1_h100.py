from teleai_pipe.common.model_manager import ModelManager
import os
from teleai_pipe.pipelines.teleai_longvideo_v1 import (
    TeleAILongVideoV1,
    get_target_image_size,
)
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander
from teleai_pipe.pipelines.flashvsr import FlashVSRPipeline
from teleai_pipe.utils.video import save_video
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
import torch
import time
from loguru import logger

PPL_CONFIG = dict(
    name="teleai_i2lv_14bv1",
    model_root="/nvfile-heatstorage/model_zoo/release/video/TeleAI-I2V-10B_250830",
    negative_prompt = (
        "jittery, shaky, fliker, unstable, ultra fast motion, distorted, blurry details, " 
        "worst quality, low quality, ugly, poorly drawn legs, poorly drawn face, " 
        "deformed, mutilated, disfigured, malformed limbs, fused fingers, three legs"
    ),
    base_num_inference_steps=60,
    st_num_inference_steps=30,
    num_frames=121,
    resolution="480p", #480p, 720p, 1080p, 2k, 4k
    base_cfg_scale=6,
    st_cfg_scale=6,
    base_sigma_shift=5.5,
    st_sigma_shift=5,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="TeleAI-I2LV-14B",
    sample_solver="euler",
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_i2vlongvideo_20251027/diffusion_pytorch_model_b15e24f.bin"
    ],
    use_cfg_zero_star=True,
    sr_dit_path_list=[
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_2k_25080922/diffusion_pytorch_model_d5916c.bin"
    ],
)

VSR_PPL_CONFIG = dict(
    name="flashvsr_1.3b",
    dit_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/flashvsr_dit_streaming_dmd.safetensors",
    vae_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/TCDecoder.ckpt",
    # vae_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/Wan2.1_VAE.pth",
    sparse_ratio=2,
    seed=0,
    tiled=True,
)


def get_pipeline(parallelism=1):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
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
    pipe = TeleAILongVideoV1.from_model_manager(
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
    pipe.enable_cpu_offload()
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([VSR_PPL_CONFIG["dit_path"]])
    mm.load_models([VSR_PPL_CONFIG["vae_path"]])
    sr_pipe = FlashVSRPipeline.from_model_manager(mm, device=f"cuda:{parallelism-1}")
    sr_pipe.enable_cpu_offload()
    pipe.sr_pipe = sr_pipe
    return pipe


def run(
    pipeline,
    image,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    video_length=10,
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
        video_lenght(int): 视频时长 5s, 10s
    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    if "MultiShot" not in prompt:
        negative_prompt = "MultiShot; <Scene Cut>, " + negative_prompt
    if video_length == 10:
        st_scale = 2
    elif video_length == 20:
        st_scale = 4
    else:
        logger.error(f"video length {video_length} is not supported")
        return None
    width, height = get_target_image_size(
        image.size[0], image.size[1], resolution="480p"
    )
    video = pipeline(
        prompt=prompt,
        input_image=image,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        base_num_inference_steps=PPL_CONFIG["base_num_inference_steps"],
        st_num_inference_steps=PPL_CONFIG["st_num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        base_cfg_scale=PPL_CONFIG["base_cfg_scale"],
        st_cfg_scale=PPL_CONFIG["st_cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        height=height,
        width=width,
        base_sigma_shift=PPL_CONFIG["base_sigma_shift"],
        st_sigma_shift=PPL_CONFIG["st_sigma_shift"],
        st_scale=st_scale,
        rand_device="cpu",
        use_cfg_zero_star=PPL_CONFIG["use_cfg_zero_star"],
        sr_scale=1,
    )
    if resolution != "480p":
        target_width, target_height = get_target_image_size(
            image.size[0], image.size[1], resolution=resolution
        )
        scale = target_width / width
        video = pipeline.sr_pipe(
            seed=VSR_PPL_CONFIG["seed"],
            LQ_video=video,
            is_full_block=False,
            if_buffer=False,
            sparse_ratio=VSR_PPL_CONFIG["sparse_ratio"],
            kv_ratio=3.0,
            local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
            color_fix=True,
            scale=scale,
            tiled=VSR_PPL_CONFIG["tiled"],
            time_tile_decode=True if len(video) > 81 else False,
        )

    return video


if __name__ == "__main__":
    prompt_config = dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="en",
    )
    prompt_expander = OpenAIPromptExpander(
        model_name=prompt_config["name"], address=prompt_config["url"], is_vl=True
    )
    pipe = get_pipeline(4)
    prompt = "The camera slowly zooms in, gradually focusing on an astronaut riding a glowing horse. The astronaut remains seated as the horse steadily walks across the waves."
    image = Image.open(
        "/nvfile-heatstorage/AIGC_H100/myk/assets/teletest/astronaut.png"
    ).convert("RGB")
    # origin_prompt = prompt
    # prompt = prompt_expander(
    #    prompt,
    #    image=image,
    #    seed=100,
    #    tar_lang=prompt_config["target_lan"],
    #
    # ).prompt
    start = time.time()
    video = run(pipe, image, prompt, video_length=20)
    print(f"generate video cost {time.time()-start}s")
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=24,
        quality=6,
    )
    print("finish save video")
    del pipe
