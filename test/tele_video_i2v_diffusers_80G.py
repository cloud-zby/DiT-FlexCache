from I2VPrompt import PROMPT_CONFIGS
from typing import Dict, List, Tuple, Optional
from functools import partial
from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import json
from teleai_pipe.pipelines.teleai_video_v1.pipeline_diffusers import VideoPipeline
import torch
from diffusers.utils import export_to_video
import torch.multiprocessing as mp

MODEL_TYPE_MAPPING = {16: "t2v", 32: "i2v", 33: "i2vhy"}
DEFAULT_CONFIG = {
    "dst_size": (1280, 720),
    "guidance_scale": 3.0,
    "embedded_guidance_scale": 1.0,
    "save_fps": 15,
    "num_frames": 49,
    "inference_steps": 50,
    "base_model": "/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b",
}
CHECKPOINT_PATH = "/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_720p_33_45_dp160_nogym/models/checkpoint_epoch_1_step_7000/transformer"
SAVEDIR = "i2vhy_720p_33_45_dp160_nogym_step7000"
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
        base_model: str = DEFAULT_CONFIG["base_model"],
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.dst_size = dst_size
        self.guidance_scale = guidance_scale
        self.embedded_guidance_scale = embedded_guidance_scale
        self.save_fps = save_fps
        self.num_frames = num_frames
        self.inference_steps = inference_steps
        self.base_model = base_model


def get_parent_path(path: str, level: int = 3) -> Path:
    """获取指定层级的父目录"""
    return Path(path).resolve().parents[level - 1]


def generate_video_filename(result_name: str, config: InferenceConfig) -> str:
    """生成视频文件名"""
    width, height = config.dst_size
    base_name = f"{result_name}_{height}p_f{config.num_frames}_s{config.inference_steps}_fps{config.save_fps}_g{config.guidance_scale}_eg{config.embedded_guidance_scale}_ntc"
    return f"{base_name}.mp4"


def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    """准备参考图像"""
    width, height = config.dst_size
    ref_images = [
        Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(config.num_frames)
    ]

    for frame_idx, img_path in config.ref_images.items():
        if frame_idx >= config.num_frames:
            raise ValueError(f"参考帧索引 {frame_idx} 超过总帧数 {config.num_frames}")

        original_img = Image.open(img_path).convert("RGB")
        resized_img = F.resize(
            original_img, (height, width), InterpolationMode.BILINEAR
        )
        ref_images[frame_idx] = resized_img

    return ref_images


def load_pipeline(
    model_path: str, config: InferenceConfig, device: torch.device
) -> VideoPipeline:
    """加载模型管道"""
    config_path = get_parent_path(model_path) / "config.json"
    with open(config_path) as f:
        in_channels = json.load(f)["models"]["transformer"]["in_channels"]

    model_type = MODEL_TYPE_MAPPING.get(in_channels)
    if not model_type:
        raise ValueError(f"不支持的 in_channels: {in_channels}")

    pipe = VideoPipeline.from_pretrained(
        config.base_model, transformer_model_path=model_path, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_tiling()
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
    device = torch.device(f"cuda:{device_id}")

    # 分配模型处理任务
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return

    pipe = load_pipeline(CHECKPOINT_PATH, infer_task[0][1], device)  # 使用任意配置加载
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
            ref_images = prepare_reference_images(config)

            # 执行推理
            output = pipe(
                prompt=config.prompt,
                height=config.dst_size[1],
                width=config.dst_size[0],
                num_frames=config.num_frames,
                num_inference_steps=config.inference_steps,
                seed=42,
                model_type=MODEL_TYPE_MAPPING[pipe.transformer.config.in_channels],
                ref_images=ref_images,
                guidance_scale=config.guidance_scale,
                embedded_guidance_scale=config.embedded_guidance_scale,
                teacache_l1_th=0.0,
            ).frames[0]

            export_to_video(output, str(save_path), fps=config.save_fps)
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
            **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"},
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
