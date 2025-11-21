import torch

torch.cuda.set_per_process_memory_fraction(1.0, 0)
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.hunyuan_video import HunyuanVideoPipeline
from teleai_pipe.utils.video import save_video


model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16,  # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
    device="cpu",
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/text_encoder/model.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/text_encoder_2",
        "/nvfile-heatstorage/model_zoo/modelscope/HunyuanVideo/vae/pytorch_model.pt",
    ],
    torch_dtype=torch.float16,
    device="cpu",
)
# We support LoRA inference. You can use the following code to load your LoRA model.
# Example LoRA: https://civitai.com/models/1032126/walking-animation-hunyuan-video
# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)

# Text-to-video
prompt = "kxsr, full body, no crop. A girl is walking. CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
video = pipe(
    prompt,
    seed=1,
    height=512,
    width=384,
    num_frames=129,
    num_inference_steps=18,
    tile_size=(17, 16, 16),
    tile_stride=(12, 12, 12),
)
save_video(video, "video.mp4", fps=30, quality=6)

# Video-to-video
prompt = "kxsr, full body, no crop. A girl is walking. CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, purple dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
video = pipe(
    prompt,
    seed=1,
    height=512,
    width=384,
    num_frames=129,
    num_inference_steps=18,
    tile_size=(17, 16, 16),
    tile_stride=(12, 12, 12),
    input_video=video,
    denoising_strength=0.85,
)
save_video(video, "video_edited.mp4", fps=30, quality=6)
