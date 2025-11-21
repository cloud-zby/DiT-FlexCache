# /nvfile-heatstorage/wjp_interns/xdb/MM_Exp/model/cogvideox_55k

"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import glob
import logging
from typing import Literal, Optional
import torch

from diffusers.utils import export_to_video, load_image, load_video
from teleai_pipe.pipelines.teleai_mmgen.pipeline_diffusers_cogmm import CogVideoXPipeline


import sys
# sys.path.append("/nvfile-heatstorage/wjp/code/CogVMM")
from teleai_pipe.pipelines.teleai_mmgen.utils import preprocess_video_with_resize, free_memory
from tqdm import tqdm

from pathlib import Path
import hashlib
from PIL import Image
import os
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import cv2
os.environ['TOKENIZERS_PARALLELISM']="1"

logging.basicConfig(level=logging.INFO)

dict_camera_type = {
    'rotate_right': 'c_r_90',
    'rotate_left': 'c_l_90',

    'move_forward': 't_f_90',
    'move_backward': 't_b_90',

    'move_right': 't_l_90',
    'move_left': 't_r_90',

    # 'move_up': 't_up_90',
    # 'move_down': 't_d_90',
}

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
    # 'cogvideox_55000_model': (240,320) # 应该反过来(height,width)
}

pipe = None

def encode_video(vae=None, video: torch.Tensor = None) -> torch.Tensor: 
    # shape of input video: [B, C, F, H, W]
    # vae = self.components.vae
    video = video.to(vae.device, dtype=vae.dtype) # torch.Size([1, 3, 48, 240, 320])
    # latent_dist = pipe.vae.encode(video).latent_dist # 一堆系数,mean,std,var # 显存增加最大的地方
    latent_dist = vae.encode(video).latent_dist # 一堆系数,mean,std,var # 显存增加最大的地方,增加约25G左右
    latent = latent_dist.sample() * vae.config.scaling_factor # torch.Size([1, 16, 12, 30, 40])
    # print("[INFO]: latent.shape:{latent.shape}") 
    free_memory()
    return latent

def preprocess(video_path: Path,max_num_frames,height,width,key='',target_fps=-1) -> torch.Tensor:
    return preprocess_video_with_resize(
        video_path,
        max_num_frames,
        height,
        width,
        key=key,
        target_fps=target_fps
    )

def embed_video(vae,video_cond_path,max_num_frames,height,width,key='',target_fps=-1) -> torch.Tensor: # max = 48, height =240, width=320
    frames = preprocess(video_cond_path,max_num_frames,height,width,key=key,target_fps=target_fps)
    save_frames = frames
    frames = frames.to("cuda") # 增加显存
    # Current shape of frames: [F, C, H, W]
    frames = video_transform(frames) # torch.Size([48, 3, 240, 320]) # 增加显存
    # Convert to [B, C, F, H, W]
    frames = frames.unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4).contiguous() # torch.Size([1, 3, 48, 240, 320])
    encoded_video = encode_video(vae,frames) # 前面会预处理视频，然后编码视频

    # [1, C, F, H, W] -> [C, F, H, W]
    encoded_video = encoded_video[0] # torch.Size([16, 12, 30, 40]);validation这里不需要去掉第一个维度
    # print(f"[INFO]: encoded_video.shape:{encoded_video.shape}")
    return encoded_video,save_frames

from torchvision import transforms
def video_transform(frames: torch.Tensor) -> torch.Tensor:
    __frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
    return torch.stack([__frame_transform(f) for f in frames], dim=0)


def get_pipeline(device):
    args = get_config_args() 

    model_path=args.model_path
    lora_path=args.lora_path
    lora_rank=args.lora_rank
    generate_type=args.generate_type

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        assert False
    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)

    pipe.to(device) # 移到gpu要一定的时间

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload() # slow inference
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.vae.requires_grad_(False) # !!! must false ， 否则会进入训练模式， 增加显存

    return pipe 

def get_camera_cond_dir(args, camera_type):
    cond_video_path = args.cond_video_path

    camera_type_index = dict_camera_type[camera_type]
    dir_camera_cond = f'{cond_video_path}/{camera_type_index}'
    # TODO: add index to camera_type

    return dir_camera_cond

def process_one_prompt(pipe, args, prompt, camera_cond_type_name, path_vid_save):
    if args is None:
        args = get_config_args()
    num_frames=args.num_frames
    width=args.width
    height=args.height
    # image_or_video_path=args.image_or_video_path,
    num_inference_steps=args.num_inference_steps
    guidance_scale=args.guidance_scale
    num_videos_per_prompt=args.num_videos_per_prompt
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    seed=args.seed
    fps=args.fps
    task_keys = args.task_keys
    cond_video_path = args.cond_video_path

    # convert
    camera_cond_type = dict_camera_type[camera_cond_type_name]


    mmc_args = {
        "idx_cond_modality": args.idx_cond_modality, 
        "cond_n": args.cond_n,
    }
    # print("[INFO] idx_cond_modality:",mmc_args["idx_cond_modality"])
    
    condition = []
    condition.append(os.path.join(cond_video_path, camera_cond_type,"moments",camera_cond_type+"_000.mp4"))
    condition.append(os.path.join(cond_video_path,camera_cond_type, "rays",camera_cond_type+"_000.mp4"))
    # from finetune.utils import string_to_filename
    # prompt_filename = prompt[:25]
    # # Calculate hash of reversed prompt as a unique identifier
    # reversed_prompt = prompt[::-1]
    # extension = "mp4"
    # hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5] 
    # file_name = f"{camera_cond_type}-{prompt_filename}-merge_{mmc_args['idx_cond_modality']}_seed={seed}.{extension}"
    # file_name = os.path.join(args.output_dir, file_name)
    # if os.path.exists(file_name):
    #     continue

    # os.makedirs(args.output_dir, exist_ok=True)
    if args.need_split:
        meta_dict={'modal':{}}
        meta_dict['prompt'] = prompt

    #!!! cond settting
    input_videos = []
    mmc_args['latent_cond_modal'] = []
    if mmc_args['idx_cond_modality'][0] != -1:
        for cond_path in condition:
            cal_key = '' # 计算的关键字
            # print(cond_path)
            encoded_video,save_frames = embed_video(pipe.vae,cond_path,num_frames,height,width,key=cal_key,target_fps=fps) # 期望的fps
            mmc_args['latent_cond_modal'].append(encoded_video)
            save_frames = save_frames.permute(0,2,3,1) # image: [T,C,H,W]
            input_video = np.array(save_frames).astype(np.uint8) # uint8
            input_videos.append(input_video)


    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps, # num_inference_steps
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        task_keys=task_keys,
        mmc_args=mmc_args
    ) # 增加4-5GB显存
    # print("[INFO] video_generate",video_generate)
    validation_artifacts_list = video_generate  # validation后的结果

    combined_array = []
    labels = []

    for validation_dict in validation_artifacts_list:
        key, value = list(validation_dict.items())[0]
        combined_array.append(np.array(value['frames'][0])) # 将所有的frame转为np.array类型，shape=(49,256,256,3)
        labels.append(key)

    if mmc_args['idx_cond_modality'][0] != -1: # 采用条件模态
        combined_array.extend(input_videos)
        labels.extend(['condition']*len(input_videos))

    if mmc_args['idx_cond_modality'] == 4: # 采用条件模态
        cond_n = mmc_args.get('cond_n',None)
        assert cond_n is not None, "cond_n should be set"
        for i in range(cond_n):
            combined_array.append(input_videos[i])
            key = task_keys[i+1]
            labels.append(f'cond_{key}')
    
    if True:
        # 将视频分割成对task_keys段，并直接保存
        for validation_dict in validation_artifacts_list:
            key, value = list(validation_dict.items())[0]
            if key != 'rgb':
                continue
            # file_name = f"validation-{prompt_filename}-merge_{mmc_args['idx_cond_modality']}_seed={seed}_split={key}.{extension}"
            # file_name = os.path.join(args.output_dir, file_name)
            # meta_dict['modal'][key] = file_name
            export_to_video(value['frames'][0], path_vid_save, fps=fps)
            # print("[INFO] save to ",path_vid_save, "key", key)    
        # save_metas.append(meta_dict)
        
    # file_name = os.path.join(args.output_dir, file_name)     
    # video_output = add_labels_to_video(combined_array, labels) # 将所有的模态的视频横向拼接在一起
    # video_output = [Image.fromarray(frame) for frame in video_output]

    # file_name_merge = path_vid_save.replace(".mp4",f"_merge_rgb.mp4")
    # export_to_video(combined_array[0], file_name_merge, fps=fps)
    # print("[INFO] save to ",file_name_merge)

    # return path_vid_save, file_name_merge

def generate_video(
    #  seed: int = 42,
    pipe = None,
    args = None,
):
    

   
    # n_seg = int(args.n_seg)  # 每进程处理的类别数量
    # idx_seg = int(args.idx_seg) # 第i段的索引
    # shuffle the meta_path
    # np.random.shuffle(meta_path) # !!！仅对选择的打乱顺序以重新推理 
    # meta_path = meta_path[idx_seg*n_seg:(idx_seg+1)*n_seg] # 每段的索引
    # save_metas = [] # 保存jsonl
    
    cond_video_path = args.cond_video_path
    case_path = os.listdir(cond_video_path)

    #!!! prompt setting
    with open(args.prompt_path, 'r') as f:
        prompts = f.readlines()

    for prompt in tqdm(prompts):
        for camera_type in case_path:
            process_one_prompt(args, prompt, camera_type)

    return None

def save_metadata(path_meta_data, lis_meta_data):
    print(f'Save samples: {len(lis_meta_data)}')
    with open(path_meta_data, "w") as f:
        for meta in lis_meta_data:
            f.write(json.dumps(meta) + "\n")


# 加载一个txt文件中的所有prompts
def load_prompts(prompt_path):
    with open(prompt_path, 'r') as f:
        # 使用 splitlines() 分割行并过滤掉空行
        prompts = [line for line in f.read().splitlines() if line.strip()]
    return prompts

import json
def load_metadata(metadata_path,shuffle=False,samples=2048):
    metas = []
    cnt = 0
    print(f"[INFO] Processing {metadata_path}")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                metas.append(json.loads(line))
                cnt += 1
        print(f"{metadata_path} {cnt} samples")
    except UnicodeDecodeError:
        print("文件编码错误，请检查文件的编码格式。")
    metas = metas[:samples] if samples < cnt else metas
    if shuffle:
        # set seed
        np.random.seed(0)
        np.random.shuffle(metas)
    return metas



def add_labels_to_video(video_list, labels, font_scale=0.75, thickness=1):
    """
    拼接多个视频，并在下面添加标签。
    
    :param video_list: List[np.ndarray]，多个视频，每个形状为 (T, H, W, C)
    :param labels: List[str]，每个视频对应的标签（如 'RGB', 'Depth'）
    :param font_scale: 字体大小
    :param thickness: 字体粗细
    :return: 拼接并带标签的视频，形状为 (T, H+label_height, W, C)
    """
    assert len(video_list) == len(labels), "视频数量和标签数量必须一致"

    # 获取视频帧信息（假设所有视频帧数 T 一致）
    print(f"video_list[0].shape: {video_list[0].shape}")
    T, H, W, C = video_list[0].shape

    # 如果不一致，则都为最短的 T
    T = min([video.shape[0] for video in video_list])
    video_list = [video[:T] for video in video_list]

    # 水平拼接视频
    video_concat = np.concatenate(video_list, axis=2)  # 沿着宽度拼接 (T, H, W_total, C)

    # 创建黑色区域用于标签（高度 = 50）
    label_height = 32
    label_area = np.ones((T, label_height, video_concat.shape[2], 3), dtype=np.uint8)

    # 计算每个视频占据的宽度区域
    section_width = W  # 每个视频的宽度
    for i, label in enumerate(labels):
        x_pos = i * section_width + section_width // 2 - 16  # 文本居中
        y_pos = label_height - 10  # 文字下方留一点空白
        cv2.putText(label_area[0], label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 复制第一帧的标签区域到所有帧
    for i in range(T):
        label_area[i] = label_area[0]

    # 将标签区域拼接到底部
    video_with_labels = np.concatenate([label_area,video_concat], axis=1)

    return video_with_labels

def get_config_args():
    """
    Generates a video based on the given prompt and saves it to the specified path.
    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    args = argparse.Namespace( **{
        # /nvfile-heatstorage/wjp_interns/wt/CogVMM/exps/MM_Exp/all_data_5_22_v2/pipeline/mmc_2000_pipeline
        # 'model_path': '/home/cogmm',
        # 'model_path': '/home/cogmm_v1_500',
        # 'model_path': '/home/cogmm_v1_1000',
        # 'model_path': '/home/cogmm_5_22_v2_2000',
        # 'model_path': '/nvfile-heatstorage/wjp_interns/wt/CogVMM/exps/MM_Exp/all_data_5_21_v1/pipeline/mmc_1000_pipeline',
        # "model_path": "/home/mmc_2000_pipe_5_27",
        "model_path": "/nvfile-heatstorage/model_zoo/mmgen/camcontrol2",
        'lora_path': None,
        'lora_rank': 128,
        # 'prompt_path': '/nvfile-heatstorage/wjp_interns/wt/CogsfmData/test_prompts/environment_subject_prompts.txt',
        'guidance_scale': 6.0,
        'num_inference_steps': 50,
        'num_frames': 49,
        'width': 640,
        'height': 480,
        'fps': 16,
        'num_videos_per_prompt': 1,
        'generate_type': 't2v',
        'dtype': 'bfloat16',
        'seed': 42,
        # 'output_dir': '/nvfile-heatstorage/wjp/code/teleai_pipe/temp/lab6_jp2',
        # 'meta_path': '/nvfile-heatstorage/wjp_interns/xdb/MM_Dataset/koala_test_20/lis_meta_data.jsonl',
        'need_split': False,
        'cond_n': 2,
        'samples': 2048,
        'idx_seg': 0,
        'n_seg': 1,
        'task_keys': ["rgb","depth","moments","rays"],
        'idx_cond_modality': [2,3],
        # 'cond_video_path': "/nvfile-heatstorage/wjp_interns/wt/CogsfmData/Scannet/processed_pose_videos"
        'cond_video_path': "/nvfile-heatstorage/model_zoo/mmgen/camcontrol2/processed_pose_videos"
        }
    )

    return args

if __name__ == "__main__":
    args = get_config_args()

    pipe = get_pipeline(args)
    prompts = load_prompts(args.prompt_path)
    # if mmc_args['idx_cond_modality'][0] != -1:
    #     assert len(args.cond_video_path) == len(args.idx_cond_modality), "The number of condition videos must be equal to cond_n"
    #     # assert len(args.idx_cond_modality) == mmc_args['cond_n'], "The number of condition indx must be equal to cond_n"
    #     for ind in mmc_args['idx_cond_modality']:
    #         print("[INFO] cond generate! cond modality is:",args.task_keys[ind])
    # else:
    #     print("[INFO] uncondition generate!")

    print(f"[INFO] validation width {args.width} , height {args.height}")
    print(f"[INFO] use {len(prompts)} prompt")

    generate_video(
        pipe=pipe,
        args=args
    )
