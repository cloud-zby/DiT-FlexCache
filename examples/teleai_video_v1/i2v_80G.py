import torch

# simlulate the 24G vram on h100, remove it on 4090 or A10
# torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import (
    TeleAIVideoV1Pipeline,
)
from teleai_pipe.utils.video import save_video
import time
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name

model_manager = ModelManager()
# The DiT model is loaded in bfloat16.
model_root = "/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1"
dit_model_root = "/workspace/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_720p_33_45_dp160_nogym/models/checkpoint_epoch_1_step_5000/transformer_safetensor"
dit_model_path_list = [
    os.path.join(dit_model_root, item)
    for item in os.listdir(dit_model_root)
    if item.endswith("safetensors")
]
model_manager.load_models(
    [dit_model_path_list],
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

# Enjoy!
prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
attn_impl = "sage_attn_8_8_sm90"  # or cudnn

num_frames = 81
start = time.time()
image = Image.open(f"{os.path.dirname(__file__)}/../data/wan_input.jpg")
image = Image.open(
    "/nvfile-heatstorage/zuoxin/workspace/video_example/101235-video-720_0.png"
)
prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
video = pipe(
    prompt,
    seed=0,
    tea_cache_l1_thresh=0,
    num_frames=num_frames,
    input_image=image,
    cfg_scale=6.0,
    cfg_skip_ratio=0,
    embedded_guidance=1.0,
    num_inference_steps=10,
    attn_impl=attn_impl,
    model_type="i2vhy",
)
video = pipe(
    prompt,
    seed=0,
    tea_cache_l1_thresh=0,
    num_frames=num_frames,
    input_image=image,
    cfg_scale=6.0,
    cfg_skip_ratio=0,
    embedded_guidance=1.0,
    num_inference_steps=50,
    attn_impl=attn_impl,
    model_type="i2vhy",
)
print(f"cost time {time.time()- start}")
save_video(
    video,
    os.path.join(
        os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"), get_example_video_name(__file__)
    ),
    fps=16,
    quality=9,
)
