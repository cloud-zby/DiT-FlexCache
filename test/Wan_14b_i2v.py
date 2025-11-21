from I2VPrompt import PROMPT_CONFIGS
from typing import Dict, List, Tuple, Optional
from functools import partial
from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import WanVideoPipeline
from teleai_pipe.utils.video import save_video
import torch.multiprocessing as mp

DEFAULT_CONFIG = {
    "dst_size": (1280, 720),
    "guidance_scale": 1.0,
    "save_fps": 15,
    "num_frames": 49,
    "inference_steps": 50,
}

GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]


class InferenceConfig:
    """集中管理推理配置参数"""

    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        dst_size: Tuple[int, int] = DEFAULT_CONFIG["dst_size"],
        guidance_scale: float = DEFAULT_CONFIG["guidance_scale"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        inference_steps: int = DEFAULT_CONFIG["inference_steps"],
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.dst_size = dst_size
        self.guidance_scale = guidance_scale
        self.save_fps = save_fps
        self.num_frames = num_frames
        self.inference_steps = inference_steps


def generate_video_filename(result_name: str, config: InferenceConfig) -> str:
    """生成视频文件名"""
    width, height = config.dst_size
    base_name = f"{result_name}_{height}p_f{config.num_frames}_s{config.inference_steps}_fps{config.save_fps}_g{config.guidance_scale}"
    return f"{base_name}.mp4"


def prepare_reference_images(config: InferenceConfig) -> Image.Image:
    """准备参考图像"""
    width, height = config.dst_size
    img_path = config.ref_images[0]

    original_img = Image.open(img_path).convert("RGB")
    input_image = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)

    return input_image


def load_model():
    model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )
    model_manager.load_models(
        [
            [
                f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{model_root}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{model_root}/Wan2.1_VAE.pth",
        ],
        # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    return model_manager


def load_pipeline(dit_model: ModelManager, device: str) -> WanVideoPipeline:
    """加载模型管道"""

    pipe = WanVideoPipeline.from_model_manager(
        dit_model,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    pipe.enable_vram_management(
        num_persistent_param_in_dit=0
    )  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    return pipe


def inference_worker(
    rank: int,
    world_size: int,
    inference_configs: List[Tuple[str, InferenceConfig]],
    gpu_ids: List[int],
):
    """推理工作进程（每个模型处理所有prompt配置）"""
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    # 分配模型处理任务
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return

    model = load_model()
    pipe = load_pipeline(model, device)  # 使用任意配置加载
    for test_name, config in infer_task:
        try:
            save_dir = Path(__file__).parent / "results" / "Wan"
            save_dir.mkdir(parents=True, exist_ok=True)

            save_name = generate_video_filename(test_name, config)
            save_path = save_dir / save_name

            if save_path.exists():
                print(f"[Rank {rank}] 跳过已存在文件: {save_path}")
                continue

            # 准备参考图像
            input_image = prepare_reference_images(config)

            # 执行推理
            output = pipe(
                prompt=config.prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                height=config.dst_size[1],
                width=config.dst_size[0],
                num_frames=config.num_frames,
                num_inference_steps=config.inference_steps,
                seed=42,
                input_image=input_image,
                cfg_scale=config.guidance_scale,
                tiled=True,
            )

            save_video(output, str(save_path), fps=config.save_fps, quality=6)
            print(f"[Rank {rank}] 保存成功: {save_path}")

        except Exception as e:
            print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {str(e)}")


def run_inference_pipeline(
    prompt_configs: Dict[str, Dict[str, str]],
    gpu_ids: List[int] = None,
):
    """运行完整推理流程（优化后版本）"""
    # GPU配置
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)

    # 准备所有推理配置
    inference_config_list = []
    for test_name, config_data in prompt_configs.items():
        inference_config = InferenceConfig(
            prompt=config_data["prompt"],
            ref_images=config_data["ref_images"],
            **{k: v for k, v in DEFAULT_CONFIG.items()},
        )
        inference_config_list.append((test_name, inference_config))

    # 启动单次多进程推理
    print(f"\n{'='*40}\n开始批量处理所有测试用例\n{'='*40}")
    mp.spawn(
        partial(
            inference_worker,
            world_size=world_size,
            inference_configs=inference_config_list,
            gpu_ids=gpu_ids,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    try:
        run_inference_pipeline(
            prompt_configs=PROMPT_CONFIGS,
            gpu_ids=GPU_IDS,
        )
    except Exception as e:
        print(f"主程序运行出错: {str(e)}")
