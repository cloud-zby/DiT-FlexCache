import torch
import sys
import os
import re
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# teleai_pipe 包在 upload/teleai_pipe/teleai_pipe 下，加入上两级目录到 sys.path
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines import wan22_video_3d
from teleai_pipe.pipelines.wan_video import get_target_image_size
from teleai_pipe.utils.video import save_video
import os
from PIL import Image
from teleai_pipe.utils.utils import get_example_video_name
from teleai_pipe.prompters.prompt_extend_s2v import OpenAIPromptExpander
import time
from loguru import logger
from decord import VideoReader, cpu
from teleai_pipe.utils.video import VideoData,crop_and_resize
import numpy as np
from pathlib import Path
import json
import pickle
import subprocess
import cv2
global PPL_CONFIG

PPL_CONFIG= dict(
    pipeline_cls="WanVideo22S2VPipeline",
    is_renderpkl=True,
    name="wan22_A14B_i2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    render_dir="./render_dir",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，重影，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    enable_magcache_dit_high=False,#############
    enable_magcache_dit_low=False,
    enable_prompt_enhance=True,
    num_inference_steps=12,# 12 10 4
    num_frames=141,
    n_segments=1,
    overlap=0,
    stride=2,
    resolution="720p",
    cfg_scale_high=1.5,
    cfg_scale_low=1.0,#######
    seed=42,
    tiled=False,
    attn_impl="sage_attn_8_8_sm90",
    model_type="Wan2.2-I2V-A14B",
    sigma_shift=5.0,
    boundary=0.875,
    sample_solver="unipc",
    wanx22_s2v_highmodel="/nvfile-heatstorage/human_guozz2/code/wxr/Teleai/models/epoch-0-step-2500.safetensors",
    #"/nvfile-heatstorage/human_guozz2/code/wxr/Teleai/DiffSynth-Studio/epoch-0-step-7500.safetensors",
    #"/nvfile-heatstorage/human_guozz2/code/wxr/Teleai/models/epoch-0-step-2500.safetensors", #######
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

    prompt_expander_config=dict(
        url="http://10.127.16.79:8999/v1/",
        name="Qwen2.5-VL-7B-Instruct",
        target_lan="en", ##########en
    ),
    lora_configs={
        "dit_high": [###新增
            {
                "path": "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
                "strength": 1.0,
            }
        ],
        "dit_low": [
            {
                "path":"/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
                "strength": 1.0,
            }
        ],
    },
)

#################################################################################

def get_pipeline_class(config):
    cls_name = config["pipeline_cls"]
    return getattr(wan22_video_3d, cls_name)

def get_video_info(video_path):
    """
    使用 decord 获取视频的总帧数和帧率

    Args:
        video_path (str): 视频文件路径

    Returns:
        total_frames (int): 总帧数
        fps (float): 帧率
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)          # 总帧数
    fps = vr.get_avg_fps()          # 平均帧率
    return total_frames, fps

def crop_and_resize_pil_list(frames, target_height, target_width):
    """
    对 numpy.ndarray 或 PIL.Image 列表进行裁剪和缩放，输出 PIL.Image 列表。
    """
    resized_list = []
    for frame in frames:
        # 如果是 numpy.ndarray 转成 PIL
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        # 裁剪和缩放
        if frame.height != target_height or frame.width != target_width:
            frame = crop_and_resize(frame, target_height, target_width)
        resized_list.append(frame)
    return resized_list
def split_pose_video(
    pose_path,
    n_segments=None,
    first_frame_id=0,
    segment_len=81,
    overlap=0,
    stride=1,   # 新增参数：间隔采样
):
    """
    把 pose 视频分段。
    - segment_len: int 或 List[int]
      * int: 固定长度，每段 segment_len 帧，总共 n_segments 段
      * list: 按列表中的长度依次切分，不用 n_segments
    - overlap: 段与段之间 overlap 帧重叠
    - stride: 控制间隔采样（默认1表示逐帧采样）
    - 不足时补齐到所需长度
    返回: List[List[np.ndarray]], true_total_len
    """
    vr = VideoReader(pose_path, ctx=cpu(0))
    total_len = len(vr)

    # 从 first_frame_id 开始采样，并加 stride
    frames = [vr[i].asnumpy() for i in range(first_frame_id, total_len, stride)]
    true_total_len = len(frames)
    if true_total_len < 81:
        print("len(frames) < required_len")
        segment_len = 81
        n_segments = 1
    # ---------------- 处理 segment_len 参数 ---------------- #
    if isinstance(segment_len, int):
        if n_segments is None:
            raise ValueError("当 segment_len 是 int 时，必须指定 n_segments")
        seg_lens = [segment_len] * n_segments
    elif isinstance(segment_len, (list, tuple)):
        seg_lens = list(segment_len)
        n_segments = len(seg_lens)
        if n_segments==2 and total_len - seg_lens[0] <= 0:
            n_segments = 1
            segment_len = seg_lens[0]
            seg_lens = [seg_lens[0]]  
    else:
        raise TypeError("segment_len 必须是 int 或 list[int]")

    required_len = sum(seg_lens)
    if true_total_len < min(seg_lens):
        print("len(frames) < min(segment_len)，强制补齐")
    
    # 不足时补齐
    if len(frames) < required_len:
        last_frame = frames[-1]
        frames.extend([last_frame] * (required_len - len(frames)))

    # ---------------- 切分 ---------------- #
    segments = []
    start = 0
    for seg_len in seg_lens:
        end = start + seg_len
        seg_frames = frames[start:end]
        segments.append(seg_frames)
        start = end - overlap  # 下一段从 overlap 往前开始
    del vr
    print("required_len:",required_len)
    return segments, true_total_len
 

def get_the_frame_pil(video_path,frame_id=0):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame = vr[frame_id].asnumpy()  # numpy array, RGB 格式
    return Image.fromarray(frame)

def get_pipeline(parallelism=1, with_prompt_expander=False):
    """
    Args:
        parallelism (int): 推理并行卡数目 2, 4 or 8
    """
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

    #s2v model 
    model_manager.load_new_module(
        file_path=PPL_CONFIG["wanx22_s2v_highmodel"],
        lora_alpha=1.0,is_high=True
    )
    PipelineClass = get_pipeline_class(PPL_CONFIG)
    print("PipelineClass:",PipelineClass)
    pipe = PipelineClass.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
        parallelism=parallelism,
        dit_high_compile=(parallelism <= 2),
        dit_low_compile=False,
        dit_high_cfg_parallel=(parallelism >= 2),
        dit_low_cfg_parallel=False,
        vae_parallel=False,
        vae_dtype=torch.float32,
        sample_solver=PPL_CONFIG["sample_solver"],
        lora_configs=PPL_CONFIG["lora_configs"],
    )
    pipe.enable_cpu_offload()
    if with_prompt_expander:
        prompt_config = PPL_CONFIG["prompt_expander_config"]
        prompt_expander = OpenAIPromptExpander(
            model_name=prompt_config["name"], address=prompt_config["url"], is_vl=True
        )
        return pipe, prompt_expander
    return pipe


def run_long_video(pipe, first_image_path="", raw_video_path="",pose_path="", prompt="", negative_prompt="", first_frame_id=0, pkl_path="", is_renderpkl=False,
                   n_segments=3, segment_len=81, overlap=8,stride=1,
                   save_dir="./output", seed=PPL_CONFIG["seed"],
                   resolution=PPL_CONFIG["resolution"], enable_prompt_enhance=True,action_id="2025"):
    """
    长视频推理：固定 n_segments 段，支持 overlap，并在最后裁掉补齐的冗余
    """
    if is_renderpkl:
        render_dir = PPL_CONFIG.get("render_dir", "./render_tmp")
        os.makedirs(render_dir, exist_ok=True)
    total_frames  = None
    origin_fps    = None
    os.makedirs(save_dir, exist_ok=True)
    if raw_video_path:
        total_frames, origin_fps = get_video_info(raw_video_path)
    if first_image_path:
        first_image = Image.open(first_image_path).convert("RGB")
    else:
        first_image = get_the_frame_pil(raw_video_path,frame_id=first_frame_id)

        # 保存第一帧
        first_image_path = os.path.join(render_dir, f"first_frame_{first_frame_id}.png")
        first_image.save(first_image_path)
        print(f"Saved first frame to {first_image_path}")
    if origin_fps:
        stride = 1 if int(origin_fps) == 16 else stride
    # 分段
    if is_renderpkl:
        with open(pkl_path, "rb") as f:
            pose_data = pickle.load(f)
        frame_idxs = pose_data["frame_idx"].tolist()  # 转成列表
        true_len = len(set(frame_idxs))
        pkl_name = os.path.basename(pkl_path).split(".")[0]
        temp_save_path = os.path.join(render_dir,pkl_name)
        os.makedirs(temp_save_path, exist_ok=True)
        sample_step  = 1
        if isinstance(segment_len, int):
            max_frames = n_segments * segment_len + 1
        elif isinstance(segment_len, (list, tuple)):
            seg_lens = list(segment_len)
            n_segments = len(seg_lens)
            max_frames = sum(segment_len)*n_segments
            if true_len / stride < segment_len[0]:
                n_segments = 1
                segment_len = segment_len[0]
            
        else:
            raise TypeError("segment_len 必须是 int 或 list[int]")        
        max_frames = min(max_frames,true_len)###############
        render_frames = total_frames if total_frames is not None else max_frames
        if raw_video_path == "/nvfile-heatstorage/Text2Video/data/raw_video/guangdian_4/B007成品/1080P/4735.mp4": #拼接动作
            print("拼接动捕动作渲染=====>")
            render_frames = max_frames
        render_fps = origin_fps if origin_fps is not None else 16

        #print(f"should render frames {max_frames},video has frames{total_frames}")
        
        command = (
            f"python ./teleai_pipe/utils/render3dmesh_subprocess_v1.py "
            f"--pkl_path {pkl_path} "
            f"--start_frame_id {0} "
            f"--max_frames {render_frames} "
            f"--sample_step {sample_step} "
            f"--first_frame_path {first_image_path} "
            f"--temp_save_path {temp_save_path} "
            f"--temp_save_fps {int(render_fps)}"
        )
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Render subprocess failed")

        pose_path = os.path.join(temp_save_path, f"{pkl_name}.mp4")

        cap = cv2.VideoCapture(pose_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"渲染视频总帧数 all render frames are: {total_frames} frames")

    pose_segments, true_total_len = split_pose_video(
        pose_path, n_segments=n_segments, first_frame_id=first_frame_id, segment_len=segment_len, overlap=overlap,stride=stride
    )
    results = []
    cur_image = first_image
    infer_final = False
    prompt_image = None

    for seg_idx, seg_frames in enumerate(pose_segments):
        logger.info(f"==== 推理第 {seg_idx+1}/{n_segments} 段 ====")
        infer_final = (seg_idx == n_segments - 1)
        print("seg_frames len", len(seg_frames), "infer_final:", infer_final)
        num_frames = len(seg_frames)
        if infer_final:
            prompt_image = first_image
        video = run(pipe, cur_image, seg_frames, prompt, prompt_image,negative_prompt,num_frames,
                    seed, resolution, enable_prompt_enhance, infer_final)

        # 如果不是第一段，需要去掉 overlap 部分
        if seg_idx > 0 and overlap > 0:
            video = video[overlap:]

        results.extend(video)
        if overlap == 0:
            cur_image = video[-1]
        else:
            cur_image = video[-overlap]

    # 裁掉补齐冗余帧
    if len(results) > true_total_len:
        results = results[:true_total_len]

    # 构造带信息量的文件名
    pose_name = Path(pose_path).stem  # pose 文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    
    if len(results) > 0:
        last_frames = results[-1:] * 5   # 复制最后一帧 5 次
        results.extend(last_frames)
    final_filename = f"{action_id}_{len(results)}_{pose_name}_{timestamp}.mp4"
    final_path = os.path.join(save_dir, final_filename)
    save_video(results, final_path, fps=15, quality=6)
    #logger.info(f"[message] Len frames are {len(results)}")
    logger.info(f"[message] Save the whole video to {final_path}")

    return results

def run(
    pipeline,
    image,
    pose_path,
    prompt,
    prompt_image=None,
    negative_prompt="",
    num_frames =81,
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    enable_prompt_enhance: bool = True,
    infer_final: bool = False,
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
        enable_prompt_enhance(bool): 是否使用prompt enhance

    Returns:
        List[PIL.Image]: 生成的视频序列
    """
    width, height = get_target_image_size(
        image.size[0], image.size[1], resolution=resolution
    )
    print("width, height==============================>",width, height)
    is_diving=False
    words = ["diving", "diver","跳水"]

    pattern = r"\b(" + "|".join(words) + r")\b"
    if re.search(pattern, prompt):
        print("is diving")
        is_diving = True
    else:
        print("no diving")
        
    if isinstance(pose_path,str):
        control_video = VideoData(pose_path,height=height, width=width)
    else:
        control_video = crop_and_resize_pil_list(pose_path, target_height=height, target_width=width)

    if isinstance(pipeline, tuple):
        real_pipe, prompt_expander = pipeline
        if enable_prompt_enhance:
            origin_prompt = prompt
            if is_diving:
                tar_lang="en"#zh
            else:
                tar_lang = PPL_CONFIG["prompt_expander_config"]["target_lan"]

            # First, expand the original prompt
            prompt_output,is_diving = prompt_expander(
                origin_prompt,
                image=prompt_image if prompt_image is not None else image,
                seed=100,
                tar_lang=tar_lang,
            )
            expanded_prompt = prompt_output.prompt
            prompt = f"{expanded_prompt}"
            # if not is_diving:
            #     prompt = origin_prompt

            print(f"Fused original and expanded prompt -> {prompt}")
    else:
        real_pipe = pipeline
    boundary=0.90 if is_diving else PPL_CONFIG["boundary"]
    print('num_frames==========>is_diving============>',num_frames,is_diving,boundary)
    
    video = real_pipe(
        prompt=prompt,
        input_image=image,
        control_video=control_video,
        negative_prompt=f'{negative_prompt} {PPL_CONFIG["negative_prompt"]}',
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=num_frames,
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        attn_impl=PPL_CONFIG["attn_impl"],
        model_type=PPL_CONFIG["model_type"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        enable_magcache_dit_high=PPL_CONFIG["enable_magcache_dit_high"],
        enable_magcache_dit_low=PPL_CONFIG["enable_magcache_dit_low"],
        boundary=boundary,
        infer_final=infer_final
    )
    return video

def run_with_file(
    pipeline,
    time_len,
    action_id,
    raw_video_path,
    first_frame_path,
    pose_path,
    prompt,
    negative_prompt,
    seed,
    target_video_resolution,
    save_video_dir,
    enable_prompt_enhance,
    first_frame_id=0,
    pkl_path="",
    is_renderpkl=False,
    **kwargs,
):
    # 根据时长调整 segment
    if time_len == "10s":
        segment_len = 141
        n_segments = 1

    elif time_len == "5s":
        segment_len = 81
        n_segments = 1
    elif time_len == "15s":
        segment_len = [141,81]
        n_segments = 2
    else:
        raise ValueError(
            f"Unsupported time_len {time_len}, "
            "please choose either '5s' or '10s','20s."
        )
    kwargs["segment_len"] = segment_len
    kwargs["n_segments"] = n_segments
    # 如果没有指定第一帧路径，从视频提取
    if first_frame_path and os.path.exists(first_frame_path):
        first_image = Image.open(first_frame_path).convert("RGB")
    else:
        first_image = get_the_frame_pil(raw_video_path, frame_id=first_frame_id)
        # 保存到 render_dir
        render_dir = PPL_CONFIG.get("render_dir", "./render_dir")
        os.makedirs(render_dir, exist_ok=True)
        first_frame_path = os.path.join(render_dir, f"first_frame_{first_frame_id}.png")
        first_image.save(first_frame_path)
        #print(f"Saved first frame to {first_frame_path}")
    # 调用长视频推理
    video = run_long_video(
        pipeline,
        first_image_path=first_frame_path,
        raw_video_path=raw_video_path,
        pose_path=pose_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        n_segments=n_segments,
        segment_len=segment_len,
        overlap=kwargs.get("overlap", PPL_CONFIG["overlap"]),
        stride=kwargs.get("stride", PPL_CONFIG["stride"]),
        save_dir=save_video_dir,
        is_renderpkl=is_renderpkl,
        pkl_path=pkl_path,
        enable_prompt_enhance=enable_prompt_enhance,
        action_id=action_id,
    )

    logger.info(f"Save target video to {save_video_dir} dir")
    # save_video(video, save_video_path, fps=16, quality=6)
def batch_run(json_path, output_root, time_len="10s"):
    # 加载 pipeline
    pipe = get_pipeline(2, True)

    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        assets_list = json.load(f)

    logger.info(f"Loaded {len(assets_list)} video assets from {json_path}")

    for i, assets in enumerate(assets_list):
        video_id = assets["id"]
        logger.info(f"[{i+1}/{len(assets_list)}] Processing video_id={video_id}")

        # 每个视频单独建输出目录
        save_video_dir = output_root


        start = time.time()
        try:
            run_with_file(
                pipeline=pipe,
                time_len=time_len,
                action_id=assets["id"],
                raw_video_path=assets["raw_video_path"],
                first_frame_path=assets.get("first_frame_path", ""),
                pose_path=assets["3d_mesh_mp4"],
                prompt=assets["caption"]["dense_caption"][0],
                negative_prompt=PPL_CONFIG["negative_prompt"],
                seed=PPL_CONFIG["seed"],
                target_video_resolution=PPL_CONFIG["resolution"],
                save_video_dir=save_video_dir,
                enable_prompt_enhance=PPL_CONFIG["enable_prompt_enhance"],
                first_frame_id=assets.get("first_frame_id", 0),
                pkl_path=assets.get("3d_smpl_pkl", ""),
                is_renderpkl=PPL_CONFIG["is_renderpkl"],
                n_segments=PPL_CONFIG["n_segments"],
                segment_len=PPL_CONFIG["num_frames"],
                overlap=PPL_CONFIG["overlap"],
                stride=PPL_CONFIG["stride"],
            )
            logger.info(f"✔ Finished video_id={video_id} in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"✘ Failed video_id={video_id}, error: {e}")

    del pipe
    logger.info("===== All batch jobs finished =====")

# if __name__ == "__main__":
#     json_path = "/nvfile-heatstorage/AIGC_H100/lbc/database/motion_2.0/all_videos_0828.json"
#     output_root = "/nvfile-heatstorage/human_guozz2/code/wxr/Teleai/output/batch_new_unipc_0910"
#     batch_run(json_path, output_root, time_len="15s")

if __name__ == "__main__":
    pipe = get_pipeline(2, True)
    output_dir = "/nvfile-heatstorage/human_guozz2/code/wxr/Teleai/output_0910_testpkl"
    # pkl_dir = "/nvfile-heatstorage/myk/vast/teleai_pipe/teleai_pipe/teleai_pipe/test/demo1"
    #     # 遍历所有 pkl 文件
    # pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
    # pkl_files.sort()  # 保证顺序一致

    # for idx, pkl_file in enumerate(pkl_files, start=1):
    #     pkl_path = os.path.join(pkl_dir, pkl_file)

    #     assets = {
    #         "id": f"2025_{idx}",
    #         "raw_video_path": "",
    #         "first_frame_path": "/nvfile-heatstorage/human_guozz2/code/wxr/upload/teleai_pipe/examples/data/111111(3).png",
    #         "3d_mesh_mp4": "",
    #         "3d_smpl_pkl": pkl_path,
    #         "first_frame_id": 0,
    #         "prompt": "一个男人正在做运动",
    #         "negative_prompt": PPL_CONFIG["negative_prompt"],
    #         "is_renderpkl": True,
    #         "time_len": "10s"
    #     }

    #     print(f"\n===== Start processing {pkl_file} =====")
    #     start = time.time()
    #     run_with_file(
    #         pipeline=pipe,
    #         time_len=assets["time_len"],
    #         action_id=assets["id"],
    #         raw_video_path=assets["raw_video_path"],
    #         first_frame_path=assets["first_frame_path"],
    #         pose_path=assets["3d_mesh_mp4"],
    #         prompt=assets["prompt"],
    #         negative_prompt=assets["negative_prompt"],
    #         seed=PPL_CONFIG["seed"],
    #         target_video_resolution=PPL_CONFIG["resolution"],
    #         save_video_dir=output_dir,
    #         enable_prompt_enhance=PPL_CONFIG["enable_prompt_enhance"],
    #         first_frame_id=assets["first_frame_id"],
    #         pkl_path=assets["3d_smpl_pkl"],
    #         is_renderpkl=assets["is_renderpkl"],
    #         n_segments=PPL_CONFIG["n_segments"],
    #         segment_len=PPL_CONFIG["num_frames"],
    #         overlap=PPL_CONFIG["overlap"],
    #         stride=PPL_CONFIG["stride"],
    #     )
    #     print(f"Generate video cost {time.time() - start:.2f}s")
    #     print(f"===== Finish {pkl_file} =====")

    # del pipe
    # assets = {
    #     "raw_video_path": "/nvfile-heatstorage/Text2Video/new_ppl_data/ysp_sport_clip/tiaoshui/sport_tiaoshui_clip_0211/v1.0/clip_video/sport_tiaoshui_clip_0211/f000048966y_1/f000048966y_1-5_0.mp4",
    #     "first_frame_path": "",  # 如果有第一帧路径可以直接指定
    #     "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/filtered_dataset_new_cmpmp4/f000048966y_1-5_0.mp4",
    #     "3d_smpl_pkl": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/filtered_dataset_new_cmppkl/f000048966y_1-5_0.pkl",
    #     "first_frame_id": 0,
    #     "prompt": "A diver performs an acrobatic flip into a swimming pool during a competition.",
    # }
    # assets = {
    #     "id":"2025",
    #     "raw_video_path": "/nvfile-heatstorage/Text2Video/new_ppl_data/ysp_sport_clip/tiaoshui/sport_tiaoshui_clip_0211/v1.0/clip_video/sport_tiaoshui_clip_0211/f000048966y_1/f000048966y_1-5_0.mp4",
    #     "first_frame_path": "",  # 如果有第一帧路径可以直接指定
    #     "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/filtered_dataset_new_cmpmp4/f000048966y_1-5_0.mp4",
    #     "3d_smpl_pkl": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/filtered_dataset_new_cmppkl/f000048966y_1-5_0.pkl",
    #     "first_frame_id": 0,
    #     "prompt": "一个女性运动员正在跳水",
    #     "negative_prompt": PPL_CONFIG["negative_prompt"],
    #     "is_renderpkl": False,
    #     "time_len": "10s"
    # }
    assets = {
        "id":"20250620-3d-3aec3705",
        "raw_video_path": "/nvfile-heatstorage/Text2Video/new_ppl_data/ysp_sport_clip/ticao/sport_ticao_clip_0108/v1.0/clip_video/sport_ticao_clip_0108/s0000091ymh_2/s0000091ymh_2-97_0.mp4",
        "first_frame_path": "",#"/nvfile-heatstorage/zhoulx/shared_data/first_frame_img_for_t2v_wxr_0625/s0000091ymh_2-90_0_frame5.jpg",  # 如果有第一帧路径可以直接指定
        "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/ticao_tiaoshui_filter_recaption_1_cmtmp4/s0000091ymh_2-90_0.mp4",
        "3d_smpl_pkl": "/nvfile-heatstorage/human_guozz2/code/wxr/code/UniAnimate-DiT-main/ruku/ruku_data/ticao_tiaoshui_filter_recaption_1_cmtpkl/s0000091ymh_2-97_0.pkl",
        "first_frame_id": 0,
        "prompt": "indoor arena with rows of seats, some occupied by spectators, and banners indicating the event as part of the 14th National Games of China, with bright lighting highlighting the gymnast against the darker backdrop of the audience and colorful advertisements",
        "negative_prompt": PPL_CONFIG["negative_prompt"],
        "is_renderpkl": True,
        "time_len": "15s"
    }
    # assets = {
    #     "raw_video_path": "/nvfile-heatstorage/human_guozz2/code/data/zhangyanfei/s2v/processed/ysp_sport_clip/mashu/573-1-66.mp4",
    #     "first_frame_path": "",
    #     "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/data/zhangyanfei/s2v/processed/ysp_sport_clip/mashu_cmtmp4/573-1-66.mp4",
    #     "3d_smpl_pkl": "",
    #     "first_frame_id": 0,
    #     "prompt": "A rider competes in an equestrian dressage event, guiding her horse through a precise routine.",
    #     "negative_prompt": PPL_CONFIG["negative_prompt"],
    #     "is_renderpkl": False,
    #     "time_len": "10s"
    #     }
    # assets = {
    # "raw_video_path": "/nvfile-heatstorage/human_guozz2/code/data/zhangyanfei/s2v/processed/ysp_sport_clip/mashu/573-1-66.mp4",
    # "first_frame_path": "",
    # "3d_mesh_mp4": "/nvfile-heatstorage/human_guozz2/code/data/zhangyanfei/s2v/processed/ysp_sport_clip/mashu_cmtmp4/573-1-66.mp4",
    # "3d_smpl_pkl": "",
    # "first_frame_id": 0,
    # "prompt": "A rider competes in an equestrian dressage event, guiding her horse through a precise routine.",
    # }
    #time_len="10s"

    start = time.time()
    run_with_file(
        pipeline=pipe,
        time_len=assets["time_len"],
        action_id=assets["id"],
        raw_video_path=assets["raw_video_path"],
        first_frame_path=assets["first_frame_path"],
        pose_path=assets["3d_mesh_mp4"],
        prompt=assets["prompt"],
        negative_prompt=PPL_CONFIG["negative_prompt"],
        seed=PPL_CONFIG["seed"],
        target_video_resolution=PPL_CONFIG["resolution"],
        save_video_dir=output_dir,
        enable_prompt_enhance=PPL_CONFIG["enable_prompt_enhance"],
        first_frame_id=assets["first_frame_id"],
        pkl_path=assets["3d_smpl_pkl"],
        is_renderpkl=assets["is_renderpkl"],
        n_segments=PPL_CONFIG["n_segments"],
        segment_len=PPL_CONFIG["num_frames"],
        overlap=PPL_CONFIG["overlap"],
        stride=PPL_CONFIG["stride"],
    )
    print(f"Generate video cost {time.time() - start:.2f}s")
    print("===== Finish Generate Video =====")
    del pipe
