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

PPL_CONFIG = dict(
    name="wan22_14b_i2v_official",
    num_inference_steps=4,
    num_frames=81,
    cfg_scale=(1, 1),
    checkpoint_dir="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    sigma_shift=5,
    sample_solver="euler",
    seed=42,
    lora_dir="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/",
)


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def get_pipeline():
    cfg = WAN_CONFIGS["i2v-A14B"]
    pipeline = WanI2V(
        config=cfg,
        checkpoint_dir=PPL_CONFIG["checkpoint_dir"],
        convert_model_dtype=True,
        lora_dir=PPL_CONFIG["lora_dir"],
    )
    return pipeline


def run(pipline: WanI2V, image, prompt, seed=PPL_CONFIG["seed"]):
    video = pipline.generate(
        prompt,
        image,
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
    image = Image.open(
        f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png"
    ).convert("RGB")
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    start = time.time()
    video = run(pipe, image=image, prompt=prompt)
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
