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
from datetime import datetime
from teleai_pipe.utils.vision import image_compose_width, save_video
import os
import json
import cv2
from decord import VideoReader, cpu
import pickle
import subprocess
import shutil
import torch.distributed as dist


def get_video_fps(video_path):
    try:
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        print(f"[info] 视频帧率: {fps:.2f} FPS")
        return fps
    except Exception as e:
        print(f"[error] 读取视频失败: {e}")
        return None


def process_single_image(image, target_size, color_filled=None, output_path=None):
    def get_edge_average_color(image):
        # 将 PIL 图像转换为 NumPy 数组
        image_array = np.array(image)
        # 提取顶部和底部边缘的像素值
        top_edge = image_array[0, :, :]
        bottom_edge = image_array[-1, :, :]
        # 将顶部和底部边的像素值合并
        combined_edge = np.concatenate((top_edge, bottom_edge), axis=0)
        # 将颜色值转换为元组以便统计
        colors = [tuple(color) for color in combined_edge]
        # 统计颜色出现次数
        color_counts = {}
        for color in colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        # 选择出现频率最高的颜色作为填充颜色
        mode_color = max(color_counts, key=color_counts.get)
        return np.array(mode_color)

    def add_colored_background(image, target_size, average_color):
        # 创建纯色背景
        colored_background = Image.new("RGB", target_size, tuple(average_color))
        # 计算原图在背景中的位置
        x_offset = (target_size[0] - image.width) // 2
        y_offset = (target_size[1] - image.height) // 2
        # 将原图拷贝到背景中
        colored_background.paste(image, (x_offset, y_offset))
        return colored_background

    def resize_image(image, target_size):
        resized_image = cv2.resize(np.array(image), (target_size[0], target_size[1]))
        return Image.fromarray(resized_image)

    def crop_imgs(img, tgt_width, tgt_height):
        width, height = img.size
        original_ratio = width / height
        aspect_ratio = tgt_width / tgt_height
        left, top, right, bottom = 0, 0, width, height
        if original_ratio != aspect_ratio:
            if original_ratio > aspect_ratio:
                new_width = int(height * aspect_ratio)
                left = (width - new_width) // 2
                top = 0
                right = left + new_width
                bottom = height
            else:
                new_height = int(width / aspect_ratio)
                left = 0
                top = (height - new_height) // 2
                right = width
                bottom = top + new_height
        img = img.crop((left, top, right, bottom))
        return img

    width, height = image.size
    tgt_width, tgt_height = target_size[0], target_size[1]
    ratio_wh = True
    # 扩充图片至正方形尺寸或插值处理
    if width < height:  # 竖版图片
        print("[message] 竖版图片", width, height)
        if width / height >= 9 / 16:
            ratio_wh = True
        else:
            ratio_wh = False
        new_size = (height, height)
        if color_filled is None:
            average_color = get_edge_average_color(image)  # np.array([255,255,255])
        else:
            average_color = color_filled
        processed_image = add_colored_background(image, new_size, average_color)
        processed_image = crop_imgs(processed_image, tgt_width, tgt_height)

    else:
        print("[message] 横板图片", width, height)
        average_color = get_edge_average_color(image)  # np.array([255,255,255])
        processed_image = crop_imgs(image, tgt_width, tgt_height)  # 也变成1：1
    # width, height = processed_image.size
    if max(width, height) < min(target_size[0], target_size[1]) or min(
        width, height
    ) > max(target_size[0], target_size[1]):
        processed_image = resize_image(processed_image, target_size)
        # processed_image.save("111.jpg")
        print("[message] 极值校正", target_size, processed_image.size)
    processed_image = resize_image(processed_image, target_size)
    # processed_image.save(output_path)
    return processed_image


def extract_clip_asset(json_path, clip_index=0, save_dir="extracted_first_frames"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载 JSON 文件
    with open(json_path, "r") as f:
        data = json.load(f)

    clips = data.get("clips", [])

    if clip_index >= len(clips):
        raise IndexError(f"索引超出范围：JSON 中只有 {len(clips)} 个 clips")

    clip = clips[clip_index]

    raw_video_path = clip.get("raw_video_path")
    cmt_path = clip.get("cmt", "")
    dense_caption = clip.get("caption", {}).get("dense_caption", [])
    caption = dense_caption[0] if dense_caption else ""

    # 提取第 0 帧
    cap = cv2.VideoCapture(raw_video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"无法读取视频第0帧：{raw_video_path}")

    # 保存帧图像
    clip_id = clip.get("id", clip_index)
    frame_filename = f"clip_{clip_id}_first_frame.png"
    frame_path = os.path.join(save_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    # 构建 assets 字典
    asset = {
        "first_frame_id": 0,
        "first_frame_path": frame_path,
        "cmt_path": cmt_path,
        "caption": caption,
    }

    return asset


import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=False)
    mp.set_sharing_strategy("file_system")
    # ---------- 参数定义 ----------
    height = 720
    width = 1280
    seed = 0
    num_frames = max_frames = 81
    continue_num_frames = num_frames * 1  # 拼接生成总帧数
    parallelism = 4

    is_lora = False
    use_teacache = True
    use_mesh_mp4 = False
    sample_step = 1
    target_fps = 15  # 你保存视频的帧率
    with_condition_visualization = False

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
            f"/nvfile-heatstorage/human_guozz2/data/other/online_models/9999/model-{i:05}-of-00011.safetensors"
            for i in range(1, 12)
        ],
        lora_alpha=1.0,
    )

    if is_lora:
        # pass
        model_manager.load_lora_v3("/nvfile-heatstorage/AIGC_H100/human_guozz2/code/zhangyan/teleai_pipe_0529/Wan2.1_I2V_14B_FusionX_LoRA.safetensors", lora_alpha=1.0)
        # model_manager.load_lora_v3("/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/models/Wan-AI/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors", lora_alpha=1.0)
    
    
    pipe = WanVideo3DS2VPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        enable_vram_management=False,
        dit_fsdp=True,
        vae_parallel=True,
        parallelism=parallelism,
        # dit_device_ids = [0,1,2,3],
        compile=(parallelism <= 2),
        use_cfg_parallel=(parallelism >= 2),
    )
    pipe.enable_cpu_offload()
    # pipe = WanVideo3DS2VPipeline.from_model_manager(
    #     model_manager, torch_dtype=torch.bfloat16, device="cuda", compile=True
    # )


    ############################################################################################
    json_path = "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/data/filtered_dataset.json"

    assets = {
        "first_frame_id": 0,
        "first_frame_path": "/nvfile-heatstorage/human_guozz2/code/zhangyan/assets/八极拳小架一路-1-起势.png",
        "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/wxr/data/DATA_A800/online_dataset/comotion/video/五禽戏-13-熊晃-左.mp4",
        "3d_mesh_pkl": "/nvfile-heatstorage/human_guozz2/data/wxr/move_to_guangzhou/wushu_ticao_tiaoshui/v0.0.1/pose_patch/comotion/data/八极拳小架一路-1-起势.pkl",
        "caption": "A person is practicing Tai Chi.",
    }


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(f"work_dirs/s2v_3d/output_{timestamp}.mp4")
    results_dir.parent.mkdir(parents=True, exist_ok=True)


    # ---------- 加载视频pose图像 ----------
    ref_frame = Image.open(assets["first_frame_path"]).resize(
        (width, height), Image.LANCZOS
    )
    if use_mesh_mp4:
        pose_data = decord.VideoReader(assets["3d_mesh_mp4"])
        fps = get_video_fps(assets["3d_mesh_mp4"])
        print("fps:", fps)
        num_all_frames = len(pose_data)
        start = assets["first_frame_id"]

        # 计算采样步长
        # sample_step = max(int(round(fps / target_fps)), 1)
        sample_step = 1

        # 生成采样帧索引
        sampled_indices = np.arange(
            start, start + continue_num_frames * sample_step, sample_step
        )
        sampled_indices = sampled_indices[sampled_indices < num_all_frames]  # 边界安全
        print("sampled_indices:", sampled_indices, len(sampled_indices))
        if continue_num_frames >= len(sampled_indices):
            continue_num_frames = len(sampled_indices)  # ✅ 加这句：更新为实际有效帧数

        frames = pose_data.get_batch(sampled_indices.astype(int))
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        cmt_images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        cmt_images = F.resize(cmt_images, (height, width), InterpolationMode.BILINEAR)
        ref_pose_tmp = cmt_images[0].permute(1, 2, 0)

    else:
        from teleai_pipe.utils.render3dmesh import draw_3d_mesh

        sample_step = 1
        start = assets["first_frame_id"]
        # 计算采样步长
        # 生成采样帧索引

        with open(assets["3d_mesh_pkl"], "rb") as f:
            pose_data = pickle.load(f)

        num_all_frames = pose_data["frame_idx"].shape[0]
        print("num_all_frames:", num_all_frames)
        sampled_indices = np.arange(
            start, start + continue_num_frames * sample_step, sample_step
        )
        sampled_indices = sampled_indices[sampled_indices < num_all_frames]  # 边界安全
        if continue_num_frames >= len(sampled_indices):
            continue_num_frames = len(sampled_indices)  # ✅ 加这句：更新为实际有效帧数
        print("sampled_indices:", sampled_indices, len(sampled_indices))

        ### draw 3d mesh as sunprocess.
        pkl_path = assets["3d_mesh_pkl"]
        start_frame_id = assets["first_frame_id"]
        total_frames = continue_num_frames * sample_step
        first_frame_path = assets["first_frame_path"]
        pkl_name = pkl_path.split("/")[-1].split(".")[0]
        temp_save_path = f"./{pkl_name}"

        command = f"python ./teleai_pipe/utils/render3dmesh_subprocess.py --pkl_path {pkl_path} --start_frame_id {start_frame_id} --max_frames {total_frames} --sample_step {sample_step} --first_frame_path {first_frame_path} --temp_save_path {temp_save_path}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("ValueError")

        save_path = f"{temp_save_path}/{pkl_name}.mp4"
        pose_data = decord.VideoReader(save_path)
        # frames = pose_data.get_batch(np.arange(0, max_frames//sample_step, dtype=int))
        frames = pose_data.get_batch(np.arange(0, len(pose_data), dtype=int))
        # print("frames==============>",len(pose_data))

        # if  continue_num_frames >= len(sampled_indices):
        #     continue_num_frames = len(sampled_indices)  # ✅ 加这句：更新为实际有效帧数
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        cmt_images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        cmt_images = F.resize(cmt_images, (height, width), InterpolationMode.BILINEAR)
        ref_pose_tmp = cmt_images[0].permute(1, 2, 0)


    # 安全读取帧

    # ---------- 分段首尾拼接式生成 ----------
    final_video = []
    final_condition = []
    step = num_frames  # - 1
    cycle = 0
    total_len = cmt_images.shape[0]
    device = cmt_images.device  # 统一 device

    for start in range(0, continue_num_frames, step):
        cycle += 1
        end = start + num_frames

        # 当前段落的有效帧
        segment = cmt_images[start:end]

        # 如果帧数不足 num_frames，补全黑帧
        if segment.shape[0] < num_frames:
            num_pad = num_frames - segment.shape[0]
            black_frame = torch.zeros(
                (1, 3, height, width), dtype=segment.dtype, device=device
            )
            pad_frames = black_frame.repeat(num_pad, 1, 1, 1)
            segment = torch.cat([segment, pad_frames], dim=0)

        # pose 数据 (B, C, H, W) -> (C, B, H, W)
        segment_pose = segment.permute(1, 0, 2, 3)

        # 参考图像，第一段用 ref_frame，后续用上一段生成的末帧
        input_img = ref_frame if start == 0 else final_video[-1]

        # 推理
        if is_lora:
            num_inference_steps = 8
            cfg_scale=1.0
            sigma_shift=2.0
            use_teacache = False
        else:
            num_inference_steps = 35
            cfg_scale=1.0
            sigma_shift=5.0
            use_teacache = True

        segment_video = pipe(
            prompt=assets["caption"],
            negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=input_img,
            num_inference_steps=num_inference_steps,  # 可以根据需要调整
            cfg_scale=cfg_scale,  # 1.5
            sigma_shift=sigma_shift,
            num_frames=num_frames,
            seed=seed,
            tiled=True,
            dwpose_data=segment_pose,
            random_ref_dwpose=ref_pose_tmp,
            height=height,
            width=width,
            context_size=21,
            context_overlap=4,
            tea_cache_l1_thresh=0.2 if use_teacache else 0,
            model_type="Wan2.1-I2V-14B-720P" if use_teacache else None,
            attn_impl= "flash_attn_2",  #"flash_attn_2",
            # attn_impl="sage_attn_8_8_sm90",  # "cudnn" if sage is not avaliable
        )

        # 拼接可视化图像：pose 和生成帧并排
        for i in range(num_frames):
            cond_img = to_pil_image(segment[i])
            composed = image_compose_width(
                F.resize(input_img, (height, width), InterpolationMode.BILINEAR),
                F.resize(cond_img, (height, width), InterpolationMode.BILINEAR),
            )
            final_condition.append(composed)
            final_video.append(segment_video[i])

    # ---------- 最终裁剪为有效采样帧数量 ----------
    total_valid_frames = len(sampled_indices)
    if len(final_video) > total_valid_frames:
        final_video = final_video[:total_valid_frames]
        final_condition = final_condition[:total_valid_frames]

    # ---------- 保存结果 ----------
    print("片段次数 cycle:", cycle)
    if with_condition_visualization:
        video_out = [
            image_compose_width(c, v) for c, v in zip(final_condition, final_video)
        ]
    else:
        video_out = final_video

    save_video(video_out, results_dir, fps=target_fps, quality=6)
    print(f"[message] 生成完成！{results_dir}")
    # del pipe
    ## Remove temporary directory
    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path)

    # 1. 销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()

    # 2. 主动结束所有可能残留的子进程
    for p in mp.active_children():
        p.terminate()
        p.join(timeout=2)
