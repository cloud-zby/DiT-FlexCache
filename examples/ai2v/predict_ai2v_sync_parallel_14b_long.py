import os
import sys
import cv2

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from teleai_pipe.pipelines.teleai_musical.wan_vae import AutoencoderKLWan
from teleai_pipe.pipelines.teleai_musical.wan_text_encoder import WanT5EncoderModel
from teleai_pipe.pipelines.teleai_musical.wan_image_encoder import CLIPModel
from teleai_pipe.pipelines.teleai_musical.wan_transformer3d_sync_FA1 import WanModelSyncFA1 as WanTransformer3DModel
from teleai_pipe.pipelines.teleai_musical.pipeline_wan_fun_inpaint_sync import WanFunInpaintSyncPipeline
from teleai_pipe.pipelines.teleai_musical.lora_utils import merge_lora, unmerge_lora
from teleai_pipe.pipelines.teleai_musical.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)

import multiprocessing
import torch
import torch.multiprocessing as mp
import librosa

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--cond_audio_path', type=str, help='Path to the audio file', default="/nvfile-heatstorage/AIGC_H100/human_guozz2/code/songqy/teleai_pipe/examples/data/piano_0/piano_0.MP3")
parser.add_argument('--cond_image_path', type=str, help='Path to the image file', default="/nvfile-heatstorage/AIGC_H100/human_guozz2/code/songqy/teleai_pipe/examples/data/piano_0/piano_0.jpg")
parser.add_argument('--save_video_path', type=str, help='Path to the save file', default="samples/wan-videos-fun-i2v_FA1_lora/output.mp4")
parser.add_argument('--prompt', type=str, help='text prompt', default="Playing musical instruments")
parser.add_argument('--size', type=str, help='output video resolution', default="720p")
parser.add_argument('--infer_num', type=int, help='infer num of batch, long video gen')
parser.add_argument('--no_lora', action='store_true')
args = parser.parse_args()


START_TIME=0
PROMPT=args.prompt
AUDIO_PATH=args.cond_audio_path
IMAGE_PATH=args.cond_image_path
VIDEO_LENGTH=81
ext = os.path.splitext(IMAGE_PATH)[1].lower()
# 如果不是 .jpg 或 .jpeg，就转换
if ext not in [".jpg", ".jpeg"]:
    # 打开图片
    img = Image.open(IMAGE_PATH).convert("RGB")  # 确保没有透明通道
    # 构造新的 JPG 路径
    new_path = os.path.splitext(IMAGE_PATH)[0] + ".jpg"
    # 保存为 JPG
    img.save(new_path, "JPEG")
    # 替换 IMAGE_PATH
    IMAGE_PATH = new_path



long_gen_audio_offset = int(0.08*(VIDEO_LENGTH-9) * 16000)
ori_audio_len = librosa.get_duration(filename=AUDIO_PATH)
lora_dir            = "/nvfile-heatstorage/model_zoo/ai2v/transformer_musical/lora/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
config_path         = "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/songqy/teleai_pipe/teleai_pipe/pipelines/teleai_musical/wan_civitai.yaml"
model_name          = "/nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-720P/" # NOTE: change this to your model path
trans_path          = "/nvfile-heatstorage/model_zoo/ai2v/transformer_musical/"
#lora_dir            = "/gemini/platform/public/aigc/human_guozz2/code/songqy/VideoXFun_hysver/models/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
# lora_dir            = "/gemini/platform/public/aigc/human_guozz2/code/songqy/VideoXFun_hysver/models/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
# config_path         = "/gemini/platform/public/aigc/human_guozz2/code/songqy/VideoXFun_hysver/config/wan2.1/wan_civitai.yaml"
# model_name          = "/gemini/platform/public/aigc/human_guozz2/model/Wan2.1-Fun-V1.1-14B-InP/"
# trans_path          = "/gemini/platform/public/aigc/human_guozz2/code/songqy/VideoXFun_hysver/wanx14b_audio_exp_only_crossattn_piano_multinode_FA1/checkpoint-15000/transformer"
if args.infer_num is not None:
  INFER_NUM=args.infer_num
else:
  INFER_NUM=int((ori_audio_len-6.48)/5.76+1)+1
if INFER_NUM > 1:
    INFER_NUM = 2
if ori_audio_len > 12:
    ori_audio_len=12

SAMPLE_SIZE_720 = [[720, 1680], [720, 1280], [720,1088], [720,960], [720,720], [960,720], [1088,720], [1280,720]]
SAMPLE_SIZE_480 = [[480, 1120], [480, 848], [480,720], [480,640], [480,480], [640,480], [720,480], [848,480]]
def closest_size(img_w, img_h, sizes):
    """ 找到与当前图像宽高比最接近的目标分辨率 """
    img_ratio = img_w / img_h
    best_size = None
    best_diff = float("inf")
    for h, w in sizes:
        ratio = w / h
        diff = abs(ratio - img_ratio)
        if diff < best_diff:
            best_diff = diff
            best_size = (w, h)
    return best_size

def crop_to_ratio(img, target_w, target_h):
    """ 居中裁剪到目标比例 """
    img_w, img_h = img.size
    target_ratio = target_w / target_h
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:  
        # 裁掉宽度
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        right = left + new_w
        top, bottom = 0, img_h
    else:  
        # 裁掉高度
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        bottom = top + new_h
        left, right = 0, img_w

    return img.crop((left, top, right, bottom))

def process_image(image_path, size_list=SAMPLE_SIZE_720):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    target_w, target_h = closest_size(img_w, img_h, size_list)
    cropped = crop_to_ratio(img, target_w, target_h)
    resized = cropped.resize((target_w, target_h), Image.LANCZOS)
    
    resized.save(image_path, "JPEG")
    return [target_h, target_w]

if args.size == "720p":
    SAMPLE_SIZE = process_image(IMAGE_PATH, size_list=SAMPLE_SIZE_720)
else:
    SAMPLE_SIZE = process_image(IMAGE_PATH, size_list=SAMPLE_SIZE_480)

test_examples = {
    "playing_piano3": {
    "prompt":  PROMPT,
    "ref_images": {0: IMAGE_PATH},
    "audio_path": AUDIO_PATH,
    "audio_offset": START_TIME,
    "infer_num": INFER_NUM,
    },
}



def run_test(sample_id, test_data, device):
    # GPU memory mode, which can be choosen in [model_full_load, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.
    # 
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    
    # GPU_memory_mode     = "sequential_cpu_offload"
    GPU_memory_mode     = "gpu"
    # Multi GPUs config
    # Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
    # For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
    # If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
    ulysses_degree      = 1
    ring_degree         = 1

    # Support TeaCache.
    enable_teacache     = False
    # Recommended to be set between 0.05 and 0.20. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    teacache_threshold  = 0.10
    # The number of steps to skip TeaCache at the beginning of the inference process, which can
    # reduce the impact of TeaCache on generated video quality.
    num_skip_start_steps = 5
    # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
    teacache_offload    = False

    # Riflex config
    enable_riflex       = False
    # Index of intrinsic frequency
    riflex_k            = 6
    
    # Choose the sampler in "Flow"
    sampler_name        = "Flow"

    # Load pretrained model if need
    transformer_path    = None
    vae_path            = None
    lora_path           = None

    # Other params
    sample_size         = SAMPLE_SIZE # [720, 1280] if args.size == "720p" else [480, 848]
    # sample_size         = [1280, 720]  # NOTE
    video_length        = VIDEO_LENGTH
    fps                 = 12.5

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype            = torch.bfloat16
    # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
    # validation_image_start  = "asset/1.png"
    validation_image_end    = None

    # prompts
    # prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
    negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    guidance_scale      = 4.0
    guidance_scale_audio = 4.5
    seed                = 42
    num_inference_steps = 50 if args.no_lora else 4
    lora_weight         = 0.55
    save_path           = "samples/wan-videos-fun-i2v_FA1_lora"

    k, v = sample_id, test_data
    ckpt_info = trans_path.split("/")[-2]
    image_name = os.path.basename(v['ref_images'][0])[:-4]
    audio_name = os.path.basename(v['audio_path'])[:-4]

    print(f"infer {image_name} with device: {device}......")

    # device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        trans_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=True)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintSyncPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    if ulysses_degree > 1 or ring_degree > 1:
        transformer.enable_multi_gpus_inference()

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    # coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    coefficients = None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(seed)
 
    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight)

    # for i, (k, v)in enumerate(test_examples.items()):
    #     if i != SAMPLE_ID:
    #         continue

    prompt = v["prompt"]
    validation_image_start = v["ref_images"][0]
    other_info = {
        "audio_path": v["audio_path"],
        "audio_offset": v["audio_offset"],
    }

    infer_num = v['infer_num']
    
    total_video_length = 0
    previous_sample = None
    total_sample = None
    init_audio_offset = other_info["audio_offset"]
    for i in range(infer_num):
        
        with torch.no_grad():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

            if enable_riflex:
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)


            input_video, input_video_mask, clip_image, ori_size = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)
                
                
            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                guidance_scale_audio = guidance_scale_audio,
                num_inference_steps = num_inference_steps,  # NOTE

                video      = input_video,
                mask_video   = input_video_mask,
                clip_image = clip_image,
                other_info = other_info,
                previous_sample = previous_sample,
                lora_dir = lora_dir if not args.no_lora else None,
            ).videos

        other_info["audio_offset"] = other_info["audio_offset"] + long_gen_audio_offset
        if lora_path is not None:
            pipeline = unmerge_lora(pipeline, lora_path, lora_weight)
        
        if i == 0:
            total_sample = sample[:,:,:-9]
            if infer_num == 1:
                total_sample = sample
        elif i != infer_num - 1:
            total_sample = torch.cat([total_sample, sample[:,:,:-9]], dim=2)
        else:
            total_sample = torch.cat([total_sample, sample], dim=2)
            
        previous_sample = sample[:,:,-9:]
        
        if i != infer_num - 1:
            total_video_length = total_video_length + video_length - 9
        else:
            total_video_length = total_video_length + video_length

    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if not os.path.exists(os.path.dirname(args.save_video_path)):
            os.makedirs(os.path.dirname(args.save_video_path), exist_ok=True)

        # index = len([path for path in os.listdir(save_path)]) + 1
        # prefix = str(index).zfill(8)
        prefix = f"ckpt_{ckpt_info}_image_{image_name}_audio_{audio_name}_g{guidance_scale}_ga_{guidance_scale_audio}"
        if total_video_length == 1:
            video_path = os.path.join(save_path, prefix + ".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(save_path, prefix + f"_{total_video_length}_{init_audio_offset}.mp4")
            save_videos_grid(total_sample, video_path, fps=fps)
        
        cut_audio_cmd = f"ffmpeg -i {other_info['audio_path']} -ss {init_audio_offset/16000} -t {total_video_length/fps} -c:a libmp3lame {str(video_path)[:-4]}.mp3"
        ffmpeg_cmd = f"ffmpeg -i {str(video_path)} -i {str(video_path)[:-4]}.mp3 -t {ori_audio_len} -c:a aac -map 0:v -map 1:a {args.save_video_path} -y"
        os.system(cut_audio_cmd)
        os.system(ffmpeg_cmd)
        os.system(f"rm {str(video_path)[:-4]}.mp3")
        os.system(f"rm {str(video_path)}")

    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()

# 每个进程分配一个显卡并执行任务
def run_on_gpu(rank, world_size, sample_id, test_data):
    device = torch.device(f'cuda:{rank}')  # 设置当前进程使用的GPU
    result = run_test(sample_id, test_data, device)
    # print(result)

# 初始化多进程环境
def init_process(rank, world_size, test_examples, fn, backend='nccl'):
    # 初始化分布式训练环境
    for i, (sample_id, test_data) in enumerate(test_examples.items()):
        if i == rank:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
            fn(rank, world_size, sample_id, test_data)

def main():
    num_gpus = 1  # 只能单张卡推理，使用CUDA_VISIBLE_DEVICES， 否则造成程序自锁
    world_size = num_gpus

    # 使用 torch.multiprocessing.spawn 来启动多进程
    init_process(0, 1, test_examples, run_on_gpu, backend='nccl')
    # mp.spawn(init_process,
    #             args=(world_size, test_examples, run_on_gpu),
    #             nprocs=num_gpus,  # 每个进程使用一个GPU
    #             join=True)

if __name__ == '__main__':
    main()