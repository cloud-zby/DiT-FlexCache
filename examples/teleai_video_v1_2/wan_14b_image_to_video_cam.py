import torch
from PIL import Image
import os
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.teleai_video_v1_4.pipeline import WanVideoPipeline
from teleai_pipe.utils.video import save_video

model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16,  # Image Encoder is loaded with float16
)

model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-Fun-V1.1-14B-Control-Camera/diffusion_pytorch_model.safetensors",
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_root}/Wan2.1_VAE.pth",
    ],
    # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
)
pipe.enable_cpu_offload()

start_image = Image.open(f"{os.path.dirname(__file__)}/../data/little_girl.png")
# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
prompt = "一个小女孩正在户外玩耍。她穿着一件蓝色的短袖衣和粉色的短裤，头发扎成一个可爱的辫子。她的脚上没有穿鞋，显得非常自然和随意。她正用一把红色的小铲子在泥土里挖土，似乎在进行某种有趣的活动，可能是种花或是挖掘宝藏。地上有一根长长的水管，可能是用来浇水的。背景是一片草地和一些绿色植物，阳光明媚，整个场景充满了童趣和生机。小女孩专注的表情和认真的动作让人感受到她的快乐和好奇心。"
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=start_image,
    tea_cache_l1_thresh=0.3,
    num_inference_steps=10,
    num_frames=49,
    cfg_scale=6.0,
    seed=43,
    tiled=True,
    height=720,
    width=1280,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    camera_movement_type="Pan_Left",
)
video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=start_image,
    tea_cache_l1_thresh=0.3,
    num_inference_steps=50,
    num_frames=49,
    cfg_scale=6.0,
    seed=43,
    tiled=True,
    height=720,
    width=1280,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    camera_movement_type="Pan_Left",
)
from teleai_pipe.utils.utils import get_example_video_name

save_video(video, get_example_video_name(__file__), fps=16, quality=6)
