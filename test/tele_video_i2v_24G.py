from I2VPrompt import PROMPT_CONFIGS
from typing import Dict, List, Tuple, Optional
from functools import partial
from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import torch

torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import TeleAIVideoV1Pipeline
from teleai_pipe.utils.video import save_video
import os
import torch.multiprocessing as mp

DEFAULT_CONFIG = {
    "dst_size": (1280, 720),
    "guidance_scale": 1.0,
    "embedded_guidance_scale": 6.0,
    "save_fps": 15,
    "num_frames": 49,
    "inference_steps": 50,
}
CHECKPOINT_PATH = "/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_720p_33_45_dp160_nogym/models/checkpoint_epoch_1_step_5000/transformer/diffusion_pytorch_model.bin"
SAVEDIR = "i2vhy_720p_33_45_dp160_nogym_step5000"
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]


class InferenceConfig:
    """集中管理推理配置参数"""

    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        dst_size: Tuple[int, int] = DEFAULT_CONFIG["dst_size"],
        guidance_scale: float = DEFAULT_CONFIG["guidance_scale"],
        embedded_guidance_scale: float = DEFAULT_CONFIG["embedded_guidance_scale"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        inference_steps: int = DEFAULT_CONFIG["inference_steps"],
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.dst_size = dst_size
        self.guidance_scale = guidance_scale
        self.embedded_guidance_scale = embedded_guidance_scale
        self.save_fps = save_fps
        self.num_frames = num_frames
        self.inference_steps = inference_steps


def generate_video_filename(result_name: str, config: InferenceConfig) -> str:
    """生成视频文件名"""
    width, height = config.dst_size
    base_name = f"{result_name}_{height}p_f{config.num_frames}_s{config.inference_steps}_fps{config.save_fps}_g{config.guidance_scale}_eg{config.embedded_guidance_scale}"
    return f"{base_name}.mp4"


def prepare_reference_images(config: InferenceConfig) -> Image.Image:
    """准备参考图像"""
    width, height = config.dst_size
    img_path = config.ref_images[0]

    original_img = Image.open(img_path).convert("RGB")
    input_image = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)

    return input_image


def load_model(model_path: str):
    model_manager = ModelManager()
    model_root = "/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1"
    model_manager.load_models(
        [model_path],
        torch_dtype=torch.bfloat16,  # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
        device="cpu",
    )
    model_manager.load_models(
        [
            f"{model_root}/text_encoder/model.safetensors",
            f"{model_root}/text_encoder_2",
            f"{model_root}/vae/pytorch_model.pt",
        ],
        torch_dtype=torch.float16,
        device="cpu",
    )
    return model_manager


def load_pipeline(dit_model: ModelManager, device: str) -> TeleAIVideoV1Pipeline:
    """加载模型管道"""

    pipe = TeleAIVideoV1Pipeline.from_model_manager(
        dit_model,
        torch_dtype=torch.bfloat16,
        device=device,
    )
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

    dit_model = load_model(CHECKPOINT_PATH)
    pipe = load_pipeline(dit_model, device)  # 使用任意配置加载
    for test_name, config in infer_task:
        try:
            save_dir = Path(__file__).parent / "results" / SAVEDIR
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
                height=config.dst_size[1],
                width=config.dst_size[0],
                num_frames=config.num_frames,
                num_inference_steps=config.inference_steps,
                seed=42,
                input_image=input_image,
                cfg_scale=config.guidance_scale,
                embedded_guidance=config.embedded_guidance_scale,
                tea_cache_l1_thresh=0.15,
            )

            save_video(output, str(save_path), fps=config.save_fps, quality=9)
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
