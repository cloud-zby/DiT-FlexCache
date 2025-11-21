import torch

# simlulate the 24G vram on h100, remove it on 4090 or A10
# torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import (
    TeleAIVideoV1Pipeline,
)
from teleai_pipe.utils.video import save_video
import os
import torch.multiprocessing as mp


if __name__ == "__main__":
    model_manager = ModelManager()
    # The DiT model is loaded in bfloat16.
    model_root = "/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1"
    model_manager.load_models(
        [
            "/nvfile-heatstorage/model_zoo/release/video/hy_fl2video/transformer/diffusion_pytorch_model.bin"
        ],
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
        enable_vram_management=True,
        dit_fsdp=True,
        parallelism=4,
        compile=False,
    )
    pipe.enable_cpu_offload()

    # Enjoy!
    prompt = "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door."
    attn_impl = "sage_attn_8_8_sm90"
    import time
    import os
    from PIL import Image

    first_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_F.jpg"
    )
    last_image = Image.open(
        "/nvfile-heatstorage/ljq/repos/teleai_pipe/test/compare_score/oven_L.jpg"
    )
    num_frames = 49
    start = time.time()
    video = pipe(
        prompt,
        seed=42,
        tea_cache_l1_thresh=0.15,
        num_frames=num_frames,
        ref_images=[first_image, last_image],
        cfg_scale=5.0,
        embedded_guidance=1.0,
        num_inference_steps=50,
        attn_impl=attn_impl,
        model_type="first_last",
    )
    print(f"cost time {time.time()- start}")
    from teleai_pipe.utils.utils import get_example_video_name

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
