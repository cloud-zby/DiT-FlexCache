import torch

# simlulate the 24G vram on h100, remove it on 4090 or A10
torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1.pipeline import TeleAIVideoV1Pipeline
from teleai_pipe.utils.video import save_video

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
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)

# Enjoy!
prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
prompt = "On the ship, Jack and Rose were dancing on the deck, surrounded by a jubilant crowd."
attn_impl = "cudnn"
import time

start = time.time()
video = pipe(
    prompt, seed=0, tea_cache_l1_thresh=0.15, num_frames=49, attn_impl=attn_impl
)
print(f"cost time {time.time()- start}")
save_video(video, "video_dance.mp4", fps=16, quality=6)
