import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.wan_video import (
    WanVideoPipeline,
    get_target_image_size,
    prepare_sparse_attention,
)
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
import time
from teleai_pipe.models.sparse_attention import SparseAttentionParams

PPL_CONFIG = dict(
    name="wan21_14B_i2v_fusionx_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P",
    negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    tea_cache_l1_thresh=0.0,
    enable_magcache=False,
    num_inference_steps=8,
    num_frames=81,  # 69
    cfg_scale=1.0,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.1-I2V-14B-720P",
    sigma_shift=2.0,
    sample_solver="unipc",
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    lora_configs=[
        {
            # "path": "/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
            "path": "/nvfile-heatstorage/model_zoo/modelscope/models--vrgamedevgirl84--Wan14BT2VFusioniX/snapshots/52c67b464a9aa548fca3629ee8c36e67f204ee2a/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
            "strength": 1.0,
        }
    ],
    resolution="720p",
    sparse_configs={
        "sparse_impl": "radial",  # ""
        "decay_factor": 0.2,
        "dense_timestep": 3,
        "dense_block": 1,
    },
)


def get_pipeline(parallelism=1):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
    model_root = PPL_CONFIG["model_root"]
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )
    model_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.float32,  # VAE is loaded with float32
    )
    model_manager.load_models(
        [PPL_CONFIG["dit_path_list"]],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    model_manager.load_models(
        [
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        dit_fsdp=False,
        parallelism=parallelism,
        compile=(parallelism <= 2),
        use_cfg_parallel=False, #(parallelism >= 2),
        vae_parallel=True,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
        lora_configs=PPL_CONFIG["lora_configs"],
    )
    pipe.enable_cpu_offload()
    return pipe


def run(
    pipeline,
    image,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
):
    """
        使用视频生成模型将静态图像转换为视频序列。
    Args:
        pipeline (VideoGenerationPipeline): 预加载的视频生成管道对象
        image (PIL.Image/ndarray): 输入图像，分辨率需匹配height/width参数
        prompt (str): 正向引导文本提示词
        negative_prompt (str, optional): 负面引导提示词，将与基础负面提示合并。默认为空
        seed (int, optional): 随机种子。默认为42
        resolution(str): 分辨率 such as 720p, 480p

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    sparse_config = PPL_CONFIG["sparse_configs"]
    # sparse parmas
    sparse_params = None
    width, height = get_target_image_size(
        image.size[0], image.size[1], resolution=resolution
    )
    print(f"raw image size is {width} {height}")
    if sparse_config.get("sparse_impl"):
        mask_map, new_size = prepare_sparse_attention(
            width, height, PPL_CONFIG["num_frames"], resolution
        )
        if mask_map is not None:
            sparse_params = SparseAttentionParams(
                sparse_impl=sparse_config.get("sparse_impl"),
                mask_map=mask_map,
                decay_factor=sparse_config.get("decay_factor"),
                dense_timestep=sparse_config.get("dense_timestep"),
                dense_block=sparse_config.get("dense_block"),
            )
            width, height = new_size
            print(f"new image size is {width} {height}")

    video = pipeline(
        prompt=prompt,
        input_image=image,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        tea_cache_l1_thresh=PPL_CONFIG["tea_cache_l1_thresh"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        cfg_skip_ratio=PPL_CONFIG["cfg_skip_ratio"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache=PPL_CONFIG["enable_magcache"],
        sparse_params=sparse_params,
    )
    return video


if __name__ == "__main__":
    pipe = get_pipeline(4)
    image = Image.open(
        f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png"
        # "/nvfile-heatstorage/AIGC_H100/benchmark/video/i2v_150/first_frame/i2v_frame0_150/09_frame0.png"
    ).convert("RGB")
    prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    # prompt = "手持镜头拍摄，视频捕捉到一株蒲公英种子头在湛蓝的天空下。随着镜头聚焦于那蓬松的白色种子，一阵微风拂过，一些种子缓缓飘离，轻盈地飞向空中。镜头跟随种子的缓慢上升轨迹，捕捉到它们优雅的舞动，营造出一种宁静而梦幻的氛围，象征着自由与新的开始。"
    start = time.time()
    video = run(pipe, image, prompt)
    print(f"generate video cost {time.time()-start}s")
    save_video(
        video,
        os.path.join(
            os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./"),
            get_example_video_name(__file__),
        ),
        fps=16,
        quality=6,
    )
    print("finish save video")
    del pipe
