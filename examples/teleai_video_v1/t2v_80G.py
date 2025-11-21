import torch
import time
import os

# simlulate the 24G vram on h100, remove it on 4090 or A10
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import TeleAIVideoV1Pipeline
from teleai_pipe.utils.video import save_video
from teleai_pipe.utils.utils import get_example_video_name


def get_pipeline():
    model_manager = ModelManager()
    # The DiT model is loaded in bfloat16.
    model_root = "/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1"
    model_manager.load_models(
        [f"{model_root}/teleai_video_v1_t2v.pt"],
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
        compile=True,
        enable_vram_management=False,
    )
    pipe.enable_cpu_offload()
    return pipe


def run(pipeline, prompt, height, width):
    attn_impl = "sage_attn_8_8_sm90"
    num_frames = 81
    start = time.time()
    video = pipeline(
        prompt,
        seed=42,
        tea_cache_l1_thresh=0.15,
        num_frames=num_frames,
        attn_impl=attn_impl,
        cfg_scale=1.0,
        embedded_guidance=6.0,
        width=width,
        height=height,
    )
    print(f"cost time {time.time()- start}")
    return video


if __name__ == "__main__":
    prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
    pipe = get_pipeline()
    attn_impl = "sage_attn_8_8_sm90"
    video = pipe(
        prompt,
        seed=42,
        tea_cache_l1_thresh=0.15,
        num_frames=81,
        cfg_scale=1.0,
        embedded_guidance=6.0,
        num_inference_steps=10,
        attn_impl=attn_impl,
        width=1280,
        height=720,
    )
    video = run(pipe, prompt, 720, 1280)

    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
