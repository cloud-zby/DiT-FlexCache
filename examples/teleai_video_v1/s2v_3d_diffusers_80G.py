from teleai_pipe.pipelines.wan_s2v_3d import WanVideo3DS2VPipeline
from teleai_pipe.common.model_manager import ModelManager
import torch
from torchvision.transforms import InterpolationMode, functional as F
from PIL import Image
import numpy as np
import pickle
import decord
from torchvision.transforms.functional import to_pil_image
import torchvision
from pathlib import Path

from teleai_pipe.utils.vision import image_compose_width, save_video

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ],
    torch_dtype=torch.float32,  # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)

model_manager.load_s2v_3d(
    [
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00001-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00002-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00003-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00004-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00005-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00006-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00007-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00008-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00009-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00010-of-00011.safetensors",
        "/nvfile-heatstorage/human_guozz2/code/zhangyan/UniAnimate-DiT-main/models_out_lora_217/lightning_logs/version_1/checkpoints/epoch=3-step=8600.ckpt/output_dir/model-00011-of-00011.safetensors",
    ],
    lora_alpha=1.0,
)

pipe = WanVideo3DS2VPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
)
pipe.enable_cpu_offload()


# define hight and width
height = 720  # 480
width = 1280  # 720
seed = 0
max_frames = 81
sample_step = 2
use_teacache = True
use_mesh_mp4 = False

assets = {
    "first_frame_id": 10,
    "first_frame_path": "/nvfile-heatstorage/human_guozz2/code/zhangyan/assets/八极拳小架一路-1-起势.png",
    "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/wxr/data/DATA_A800/sport_filter/comotion/video/八极拳小架一路-1-起势.mp4",
    "3d_mesh_pkl": "/nvfile-heatstorage/human_guozz2/data/wxr/move_to_guangzhou/wushu_ticao_tiaoshui/v0.0.1/pose_patch/comotion/data/八极拳小架一路-1-起势.pkl",
    "caption": "The person begins by walking towards the center of the room, then raises their arms in a fluid motion, bringing their hands together in front of their chest. They continue to move their arms in a circular motion, maintaining a steady pace. The movements are graceful and controlled, suggesting a form of martial arts or qigong practice. The room's decor, including the plants and the traditional door, adds to the serene and disciplined atmosphere of the scene. The overall style of the video is calm and focused, emphasizing the precision and elegance of the movements.",
}


results_dir = Path("work_dirs/s2v_3d/test.mp4")
results_dir.parent.mkdir(parents=True, exist_ok=True)

ref_frame = Image.open(assets["first_frame_path"]).resize(
    (width, height), Image.LANCZOS
)
if use_mesh_mp4:
    pose_data = decord.VideoReader(assets["3d_mesh_mp4"])
    frames = pose_data.get_batch(
        np.arange(
            assets["first_frame_id"],
            assets["first_frame_id"] + max_frames * 2,
            sample_step,
            dtype=int,
        )
    )
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    cmt_images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    cmt_images = F.resize(cmt_images, (height, width), InterpolationMode.BILINEAR)
    ref_pose_tmp = cmt_images[0].permute(1, 2, 0)
else:
    from teleai_pipe.utils.render3dmesh import draw_3d_mesh

    frames = draw_3d_mesh(
        assets["3d_mesh_pkl"],
        assets["first_frame_id"],
        max_frames * 2,
        sample_step,
        assets["first_frame_path"],
    )
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    cmt_images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    cmt_images = F.resize(cmt_images, (height, width), InterpolationMode.BILINEAR)
    ref_pose_tmp = cmt_images[0].permute(1, 2, 0)

video_out_condition = []
for ii in range(cmt_images.shape[0]):
    ss = to_pil_image(cmt_images[ii])
    video_out_condition.append(
        image_compose_width(
            ref_frame,
            torchvision.transforms.functional.resize(
                ss,
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
        )
    )

# # Image-to-video
video = pipe(
    prompt=assets["caption"],
    negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=ref_frame,
    num_inference_steps=40,
    cfg_scale=1.5,  # slow
    # cfg_scale=1.0,  # fast
    num_frames=max_frames,
    seed=seed,
    tiled=True,
    dwpose_data=cmt_images.permute(1, 0, 2, 3),
    random_ref_dwpose=ref_pose_tmp,
    height=height,
    width=width,
    context_size=21,
    context_overlap=4,  # larger than 2, (context_overlap-2)*4 frames are overlapped
    tea_cache_l1_thresh=0.2 if use_teacache else 0,
    model_type="Wan2.1-I2V-14B-720P" if use_teacache else None,
    attn_impl="sage_attn_8_8_sm90",  # "cudnn" if sage is not avaliable
)

video_out = []
for ii in range(len(video)):
    ss = video[ii]
    video_out.append(image_compose_width(video_out_condition[ii], ss))
save_video(video_out, results_dir, fps=16, quality=6)
