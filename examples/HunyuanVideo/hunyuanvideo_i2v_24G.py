import torch
import os

# torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.hunyuan_video import HunyuanVideoPipeline
from teleai_pipe.utils.video import save_video
from PIL import Image


model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/pretrained/HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16,  # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
    device="cpu",
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/text_encoder/model.safetensors",
        "/nvfile-heatstorage/model_zoo/pretrained/HunyuanVideo-I2V/hunyuan-video-i2v-720p/text_encoder_2",
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/vae/pytorch_model.pt",
    ],
    torch_dtype=torch.float16,
    device="cpu",
)

# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda",
    enable_vram_management=True,
)

i2v_resolution = "720p"
images = [
    Image.open(f"{os.path.dirname(__file__)}/../data/wan_input.jpg").convert("RGB")
]
# Image-to-video
prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
video = pipe(
    prompt,
    input_images=images,
    num_inference_steps=50,
    seed=0,
    i2v_resolution=i2v_resolution,
    tea_cache_l1_thresh=0.15,
)
save_video(video, f"video_{i2v_resolution}_low_vram.mp4", fps=30, quality=6)
