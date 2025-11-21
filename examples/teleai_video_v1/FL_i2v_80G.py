import torch

# simlulate the 24G vram on h100, remove it on 4090 or A10
# torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import (
    TeleAIVideoV1Pipeline,
)
from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.utils import get_example_video_name
import os
from PIL import Image

PPL_CONFIG = dict(
    name="teleai_video_v1_fl2v",
    model_root="/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1",
    transformer_path="/nvfile-heatstorage/model_zoo/release/video/hy_fl2video/transformer/diffusion_pytorch_model.bin",
    tea_cache_l1_thresh=0.15,
    num_inference_steps=50,
    embedded_guidance=1,
    num_frames=49,
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
    seed=42,
    attn_impl="sage_attn_8_8_sm90",
    model_type="first_last",
)


def get_pipeline():
    model_manager = ModelManager()
    # The DiT model is loaded in bfloat16.
    model_root = PPL_CONFIG["model_root"]
    model_manager.load_models(
        [PPL_CONFIG["transformer_path"]],
        torch_dtype=torch.bfloat16,  # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
        device="cpu",
    )

    # The other modules are loaded in float16.
    model_manager.load_models(
        [
            f"{model_root}/text_encoder/model.safetensors",
            f"{model_root}/text_encoder_2",
            f"{model_root}/vae/pytorch_model.pt",
        ],
        torch_dtype=torch.float16,
        device="cpu",
    )

    # We support LoRA inference. You can use the following code to load your LoRA model.
    # model_manager.load_lora("models/lora/xxx.safetensors", lora_alpha=1.0)

    # The computation device is "cuda".
    pipe = TeleAIVideoV1Pipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        enable_vram_management=False,
        compile=True,
    )
    pipe.enable_cpu_offload()
    return pipe


def run(pipeline, first_image, last_image, prompt, height, width):
    video = pipeline(
        prompt,
        seed=PPL_CONFIG["seed"],
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_frames=PPL_CONFIG["num_frames"],
        ref_images=[first_image, last_image],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        embedded_guidance=PPL_CONFIG["embedded_guidance"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
    )
    return video


if __name__ == "__main__":
    # Enjoy!
    pipe = get_pipeline()
    prompt = "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door."
    first_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_F.jpg"
    )
    last_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_L.jpg"
    )
    video = run(
        pipeline=pipe,
        first_image=first_image,
        last_image=last_image,
        prompt=prompt,
        height=720,
        width=1280,
    )
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
