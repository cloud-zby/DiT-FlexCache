from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS
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

PPL_CONFIG = dict(
    name="wan21_14b_t2v_official",
    num_inference_steps=50,
    num_frames=81,
    cfg_scale=5.0,
    sigma_shift=5.0,
    sample_solver="unipc",
    checkpoint_dir="/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B",
    seed=42,
)


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def get_pipeline():
    cfg = WAN_CONFIGS["t2v-14B"]
    pipeline = WanT2V(
        config=cfg,
        checkpoint_dir=PPL_CONFIG["checkpoint_dir"],
    )
    return pipeline


def run(
    pipline: WanT2V,
    prompt,
    aspect_ratio: str = "16:9",
    seed=PPL_CONFIG["seed"],
    resolution="720p",
):
    if aspect_ratio not in SUPPORTED_ASPECT_RATION_TO_SIZE_MAP_720P:
        logger.error(f"aspect ratio {aspect_ratio} is not supported")
        return None
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)

    video = pipline.generate(
        prompt,
        size=(width, height),
        frame_num=PPL_CONFIG["num_frames"],
        shift=PPL_CONFIG["sigma_shift"],
        sample_solver=PPL_CONFIG["sample_solver"],
        sampling_steps=PPL_CONFIG["num_inference_steps"],
        guide_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
    )
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
