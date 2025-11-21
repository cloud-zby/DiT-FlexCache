import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_sr import TeleVideoSRPipeline
from teleai_pipe.utils.video import save_video, VideoData
import os
from teleai_pipe.utils.utils import get_example_video_name
import time

PPL_CONFIG = dict(
    name="teleai_sr_1.3bv1",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=10,
    cfg_scale=5.5,
    timestep_range=(0, 500),
    sigma_shift=5,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="TeleAI-SR-1.3B",
    sample_solver="euler",
    dit_path_list=[
        # "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_25080910/diffusion_pytorch_model_729b2.bin"
        "/nvfile-heatstorage/model_zoo/release/video/teleai_video_sr_2k_25080922/diffusion_pytorch_model_d5916c.bin"
    ],
    enable_magcache=False,
    num_frames=81,
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
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )
    model_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,  # VAE is loaded with float32
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
    pipe = TeleVideoSRPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
        vae_parallel=parallelism >= 2,
        vae_dtype=torch.bfloat16,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.enable_cpu_offload()
    return pipe


def run(
    pipeline,
    input_video,
    prompt,
    negative_prompt="",
):
    """
        使用视频生成模型对视频进行超分。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        image (PIL.Image/ndarray): 输入图像，分辨率需匹配height/width参数
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    video = pipeline(
        prompt=prompt,
        input_video=input_video,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        sigma_shift=PPL_CONFIG["sigma_shift"],
        timestep_range=PPL_CONFIG["timestep_range"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
        tile_size=(242, 242),
        tile_stride=(162, 158),
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(2)
    video_path = f"{os.path.dirname(__file__)}/../data/dag.mp4"
    video_path = "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/teleai_hr_video_v1_teleai_video_10b_image_to_hr_video_h100.mp4"
    video = VideoData(video_path)
    input_video = [video[i] for i in range(PPL_CONFIG["num_frames"])]
    prompt = "high quality;masterpiece;cinematic;ultra high resolution, A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    video = run(pipe, input_video, prompt)
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
