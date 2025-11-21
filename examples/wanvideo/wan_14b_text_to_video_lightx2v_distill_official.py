from lightx2v.utils.set_config import set_config, print_config
from PIL import Image
from einops import rearrange
import os
from teleai_pipe.utils.video import save_video
from PIL import Image
import time
import numpy as np
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.pipelines.wan_video import get_target_video_size_from_ratio
from loguru import logger
from lightx2v.utils.utils import seed_all
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.wan.wan_runner import WanRunner, Wan22MoeRunner
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from easydict import EasyDict
import torch

PPL_CONFIG = dict(
    name="wan21_14b_t2v_lightx2v_distill_official",
    model_path="/nvfile-heatstorage/model_zoo/huggingface/models--lightx2v--Wan2.1-T2V-14B-StepDistill-CfgDistill/snapshots/d4b89b472ccdbf7e1fc57761fe2ce67c7a8a6d0b/",
    config_json="/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/LightX2V/configs/distill/wan_t2v_distill_8step_cfg.json",
    seed=42,
    model_cls="wan2.1_distill",
    task="t2v",
    use_prompt_enhancer=False,
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
)


def tensor2video(frames):
    frames = (frames * 255).to(torch.uint8).cpu().numpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def get_pipeline():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DTYPE"] = "BF16"
    os.environ["ENABLE_PROFILING_DEBUG"] = "true"
    os.environ["ENABLE_GRAPH_MODE"] = "false"

    config = set_config(EasyDict(PPL_CONFIG))
    print_config(config)
    seed_all(config.seed)
    runner = RUNNER_REGISTER[config.model_cls](config)
    runner.init_modules()
    return runner


def run(
    pipeline,
    prompt,
    aspect_ratio: str = "16:9",
    seed=PPL_CONFIG["seed"],
    resolution="720p",
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DTYPE"] = "BF16"
    os.environ["ENABLE_PROFILING_DEBUG"] = "true"
    os.environ["ENABLE_GRAPH_MODE"] = "false"
    pipeline.config.prompt = prompt
    pipeline.config.seed = seed
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
    pipeline.config.target_height = height
    pipeline.config.target_width = width
    video, _ = pipeline.run_pipeline(save_video=False)
    return tensor2video(video)


if __name__ == "__main__":
    pipe = get_pipeline()
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    video = run(pipe, prompt=prompt)
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
