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


if __name__ == "__main__":
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

    # The computation device is "cuda".
    pipe = TeleAIVideoV1Pipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        enable_vram_management=True,
        dit_fsdp=True,
        parallelism=4,
        compile=False,
    )

    prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
    prompt = "On the ship, Jack and Rose were dancing on the deck, surrounded by a jubilant crowd."
    attn_impl = "sage_attn_8_8_sm90"  # "sage_attn_8_8" on 4090
    import time
    import os

    num_frames = 81
    start = time.time()
    video = pipe(
        prompt,
        seed=0,
        tea_cache_l1_thresh=0.15,
        num_frames=num_frames,
        cfg_scale=1.0,
        embedded_guidance=6.0,
        num_inference_steps=50,
        attn_impl=attn_impl,
        model_type="t2v",
    )
    print(f"cost time {time.time()- start}")
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=9,
    )
    print("finish save video")
    del pipe
