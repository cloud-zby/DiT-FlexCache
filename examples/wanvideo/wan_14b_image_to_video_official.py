from wan.image2video import WanI2V
from wan.configs import WAN_CONFIGS
from PIL import Image
from einops import rearrange
import os
from teleai_pipe.utils.video import save_video
from PIL import Image
import time
import numpy as np
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.pipelines.wan_video import get_target_image_size

PPL_CONFIG = dict(
    name="wan21_14b_i2v_official",
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=5.0,
    checkpoint_dir="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P",
    sigma_shift=5.0,
    sample_solver="unipc",
    seed=42,
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
)


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def get_pipeline():
    cfg = WAN_CONFIGS["i2v-14B"]
    pipeline = WanI2V(
        config=cfg,
        checkpoint_dir=PPL_CONFIG["checkpoint_dir"],
    )
    return pipeline


def run(pipline: WanI2V, image, prompt, seed=PPL_CONFIG["seed"], resolution="720p"):
    video = pipline.generate(
        prompt,
        image,
        frame_num=PPL_CONFIG["num_frames"],
        shift=PPL_CONFIG["sigma_shift"],
        sample_solver=PPL_CONFIG["sample_solver"],
        sampling_steps=PPL_CONFIG["num_inference_steps"],
        guide_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        n_prompt=PPL_CONFIG["negative_prompt"],
    )
    return tensor2video(video)


if __name__ == "__main__":
    pipe = get_pipeline()
    image = Image.open(
        f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png"
    ).convert("RGB")
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    width, height = get_target_image_size(image.size[0], image.size[1])
    start = time.time()
    video = run(pipe, image=image, prompt=prompt, height=height, width=width)
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
