import gradio as gr
import argparse
import torch
import os
import gc
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan22_video import WanVideo22Pipeline
from teleai_pipe.utils.video import save_video
from loguru import logger
import datetime
import secrets  # 使用secrets模块以生成更安全的随机数
from teleai_pipe.pipelines.hunyuan_video import HunyuanVideoPipeline
from typing import List, Dict
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
from teleai_pipe.pipelines.wan_video import get_target_image_size


def generate_filename():
    # 获取当前日期，格式为YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    # 生成20位随机字符，这里使用secrets.choice从字母和数字中随机选择
    random_string = "".join(
        secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for _ in range(20)
    )
    # 组合日期和随机字符串作为文件名
    filename = f"{current_date}_{random_string}"
    return filename


wan_pipeline = None
tele_pipeline = None
hunyuan_pipeline = None
running_flag = False
from teleai_pipe.prompters.prompt_extend import OpenAIPromptExpander

prompt_config = dict(
    url="http://10.127.16.79:8999/v1/",
    name="Qwen2.5-VL-7B-Instruct",
    target_lan="zh",
)
prompt_expander = OpenAIPromptExpander(
    model_name=prompt_config["name"], address=prompt_config["url"], is_vl=True
)


def generate_extended_prompt(prompt, image):
    global running_flag
    if running_flag:
        return "server is busy, please try later"
    prompt = prompt_expander(prompt, image=image, seed=100, tar_lang="zh").prompt
    return prompt


PPL_CONFIG = dict(
    name="wan22_A14B_i2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    enable_magcache=True,
    num_inference_steps=40,
    num_frames=81,
    cfg_scale_high=3.5,
    cfg_scale_low=3.5,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=True,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.2-I2V-A14B",
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="unipc",
    dit_high_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    dit_low_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
)


def get_wan_pipeline():
    global PPL_CONFIG
    device_num = torch.cuda.device_count()
    device_num = 2 if device_num > 1 else 1
    logger.info(f"run wan on {device_num} gpus")
    global wan_pipeline
    if wan_pipeline is not None:
        return wan_pipeline

    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    # vae
    model_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.float32,
    )
    # dit high
    model_manager.load_model(
        PPL_CONFIG["dit_high_path_list"],
        torch_dtype=torch.bfloat16,
    )
    # dit low
    model_manager.load_model(
        PPL_CONFIG["dit_low_path_list"],
        torch_dtype=torch.bfloat16,
    )

    # t5
    model_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideo22Pipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=False,
        parallelism=device_num,
        compile=(device_num <= 2),
        use_cfg_parallel=(device_num >= 2),
        vae_parallel=False,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
    )
    pipe.enable_cpu_offload()
    wan_pipeline = pipe
    return pipe


def get_hunyuan_pipeline(device_id):
    global hunyuan_pipeline
    if hunyuan_pipeline is not None:
        return hunyuan_pipeline
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
        device=f"cuda:{device_id}",
        enable_vram_management=True,
    )
    hunyuan_pipeline = pipe
    return hunyuan_pipeline


def center_crop_with_ratio(image, target_aspect_ratio=16 / 9):
    """
    将图片中心裁剪为 16:9 的比例。

    参数:
        image (PIL.Image): 输入的图片对象。

    返回:
        PIL.Image: 裁剪后的图片对象。
    """
    # 获取图片的宽度和高度
    width, height = image.size

    # 计算目标裁剪区域的宽度和高度
    if width / height > target_aspect_ratio:
        # 如果图片的宽高比大于 16:9，裁剪宽度
        new_width = int(height * target_aspect_ratio)
        new_height = height
    else:
        # 如果图片的宽高比小于 16:9，裁剪高度
        new_height = int(width / target_aspect_ratio)
        new_width = width

    # 计算裁剪区域的左上角坐标
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # 执行中心裁剪
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


# 示例函数：这里你需要根据实际情况修改它以适应你的模型和视频生成逻辑
def generate_video(
    image,
    text,
    negative_prompt,
    frames,
    seed,
    output_path,
    fps,
    infer_steps,
    teacache_th,
    cfg_scale,
    sigma_shift,
    attn_impl,
    magcache,
):
    global running_flag
    global PPL_CONFIG
    if running_flag:
        return None, "server is busy, please try later"
    running_flag = True
    width, height = get_target_image_size(image.size[0], image.size[1])
    print(f"generate video with {frames} {width} {height}")
    if not output_path:
        file_name = generate_filename()
        output_path = f"/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/work_dirs/wan2_1_i2v/{file_name}.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pipe = get_wan_pipeline()
    video = pipe(
        prompt=text,
        input_image=image,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
        boundary=PPL_CONFIG["boundary"],
    )
    save_video(video, output_path, fps=fps, quality=6)
    del video
    torch.cuda.empty_cache()
    gc.collect()
    print(f"generate to {output_path}")
    running_flag = False
    return output_path, output_path


# Gradio界面设置
with gr.Blocks() as demo:
    gr.Markdown("## 视频生成器")

    with gr.Row():
        image_input = gr.Image(type="pil", label="选择一张图片", height=480)
        text_input = gr.Textbox(label="输入一段文字")
        extend_text_input = gr.Textbox(label="扩充后的提示词")
        nega_text_input = gr.Textbox(
            label="负面文字",
            value="相机晃动，色调艳丽，过曝, 细节模糊不清，字幕，风格，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合, 杂乱的背景，三条腿，背景人很多，倒着走",
            interactive=True,
        )
    with gr.Row():
        attn_impl_dropdown = gr.Dropdown(
            choices=["sage_attn_8_8_sm90", "cudnn"], label="选择attention type"
        )
        frames_input = gr.Number(
            precision=0, label="Frames Number", value=81, minimum=49, maximum=129
        )
        cfg_scale = gr.Number(precision=0, label="cfg", value=5, minimum=1, maximum=10)
        sigma_shift = gr.Number(
            precision=0, label="sigma shift", value=5, minimum=1, maximum=10
        )

        fps_dropdown = gr.Dropdown(
            choices=[16, 24], value=16, label="FPS", interactive=True
        )
        seed_input = gr.Number(precision=0, label="设置随机种子（整数）", value=42)
        infer_num_input = gr.Number(precision=0, label="infer steps", value=40)
        teacache_th_input = gr.Number(
            label="teacache", value=0.0, precision=2, minimum=0, maximum=0.4, step=0.05
        )
        magcache_cb = gr.Checkbox(label="magcache", value=False, interactive=True)
    output_path = gr.Textbox(label="视频保存路径", interactive=True)
    prompt_button = gr.Button("生成扩充prompt")
    generate_button = gr.Button("生成视频")
    video_output = gr.Video(label="生成的视频", height=480)
    info = gr.Textbox(label="結果")
    prompt_button.click(
        generate_extended_prompt, [text_input, image_input], extend_text_input
    )

    generate_button.click(
        fn=generate_video,
        inputs=[
            image_input,
            extend_text_input,
            nega_text_input,
            frames_input,
            seed_input,
            output_path,
            fps_dropdown,
            infer_num_input,
            teacache_th_input,
            cfg_scale,
            sigma_shift,
            attn_impl_dropdown,
            magcache_cb,
        ],
        outputs=[video_output, info],
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频生成器服务")
    parser.add_argument(
        "--server-port", "-p", type=int, default=8999, help="服务端口号 (默认: 8999)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, max_threads=1)
