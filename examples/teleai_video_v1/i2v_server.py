# ✨ [新建] services/video_api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from teleai_pipe.pipelines.teleai_video_v1.pipeline import (
    TeleAIVideoV1Pipeline,
)
from PIL import Image
from pathlib import Path
import os
from teleai_pipe.utils.video import save_video
from teleai_pipe.common.model_manager import ModelManager
from fastapi import FastAPI, HTTPException

torch.cuda.set_per_process_memory_fraction(0.3, 0)

app = FastAPI(title="TeleAI Video Generation API")

# 初始化管道 (单例模式)
pipe = None


@app.on_event("startup")
async def load_model():
    global pipe
    # 复用原有模型加载逻辑（需确保ModelManager已正确初始化）
    model_manager = ModelManager()

    model_root = "/nvfile-heatstorage/model_zoo/televideo/teleai_video_v1"
    dit_model_root = "/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_720p_33_45_dp160_nogym/models/checkpoint_epoch_1_step_5000/transformer_safetensor"
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

    pipe = TeleAIVideoV1Pipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        enable_vram_management=True,
        dit_fsdp=True,
        parallelism=4,
        compile=False,
    )


@app.post("/generate_video")
async def generate_video():
    prompt = "A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon. Sunlight sprinkles across the water's surface, shimmering with golden hues, adding a touch of warmth to this magnificent scene. As the camera zooms in, one can see the flag on board fluttering in the wind, symbolizing an indomitable spirit and the courage of adventure. This scene, full of power, is inspiring and uplifting, showcasing the fearlessness and perseverance when facing challenges."
    num_frames = 81
    image = Image.open(f"{os.path.dirname(__file__)}/../data/wan_input.jpg")

    video = pipe(
        prompt=prompt,
        seed=42,
        tea_cache_l1_thresh=0.15,
        num_frames=num_frames,
        input_image=image,
        cfg_scale=1.0,
        embedded_guidance=6.0,
        num_inference_steps=50,
        attn_impl="sage_attn_8_8",
        model_type="i2vhy",
    )

    output_path = "output_video.mp4"
    save_video(video, output_path, fps=16, quality=9)
    return FileResponse(output_path)


import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # 由于GPU资源限制，建议单worker
    )
