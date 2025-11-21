import os
import sys
import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import subprocess
import argparse 
from datetime import datetime
import random
import torch
import time

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from omegaconf import OmegaConf
import torch.multiprocessing as mp
from transformers import Wav2Vec2FeatureExtractor
from teleai_pipe.pipelines.wan_ai2v import WanAI2VPipeline
from teleai_pipe.pipelines.teleai_ai2v.config import SIZE_CONFIGS, WAN_CONFIGS
from teleai_pipe.pipelines.teleai_ai2v.audio_wav2vec.wav2vec2 import Wav2Vec2Model
from teleai_pipe.pipelines.teleai_ai2v.modules.utils import save_video_ffmpeg

sys.path.insert(0, os.path.join(os.getcwd(), "teleai_pipe/pipelines/teleai_ai2v/clearvoice"))
from clearvoice import ClearVoice
import json

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        choices=list(SIZE_CONFIGS.keys()),
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="Enable teacache for video generation."
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.1,
        help="Threshold for teacache."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )
    # parser.add_argument(
    #     "--lora_dir",
    #     type=str,
    #     nargs='+',
    #     default=None,
    #     help="The paths to the LoRA checkpoint files."
    # )
    # parser.add_argument(
    #     "--lora_scale",
    #     type=float,
    #     nargs='+',
    #     default=[1.2],
    #     help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    # )
    args = parser.parse_args()
    return args

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):

    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

# def _init_logging(rank):
#     # logging
#     if rank == 0:
#         # set format
#         logging.basicConfig(
#             level=print,
#             format="[%(asctime)s] %(levelname)s: %(message)s",
#             handlers=[logging.StreamHandler(stream=sys.stdout)])
#     else:
#         logging.basicConfig(level=logging.ERROR)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate, audio_seperator=None):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    if audio_seperator is None:
        human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        os.remove(raw_audio_path)
    else:
        human_audio_temp_path = os.path.join(raw_audio_path[:-4] + "_human.wav")
        output_wav = audio_seperator(input_path=raw_audio_path)
        audio_seperator.write(output_wav, output_path=human_audio_temp_path)
        human_speech_array, sr = librosa.load(human_audio_temp_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        os.remove(raw_audio_path)
        os.remove(human_audio_temp_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000, audio_seperator=None):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate, audio_seperator)
        return human_speech_array
    else:
        if audio_seperator is None:
            human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
            human_speech_array = loudness_norm(human_speech_array, sr)
        else:
            human_audio_temp_path = os.path.join(audio_path[:-4] + "_human.wav")
            output_wav = audio_seperator(input_path=audio_path)
            audio_seperator.write(output_wav, output_path=human_audio_temp_path)
            human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
            human_speech_array = loudness_norm(human_speech_array, sr)
            os.remove(human_audio_temp_path)
        return human_speech_array

def run_test(sample_id, test_data, device, rank):
    """
    [NOTE]:
    推理需要设置的输入:
    prompt: 描述视频的文本
    cond_image: 首图路径
    cond_audio: 驱动音频路径
    model_path: 模型文件路径
    """ 
    ################## 需要配置 or 需要输入的参数 ######################
    if test_data is None:
        test_data = {
                    "prompt": "一个小和尚双手合十，活泼地对着镜头介绍着什么",
                    "cond_image": "/nvfile-heatstorage/AIGC_H100/hys/teleai_pipe/work_dirs/examples/0812/小和尚.png",
                    "cond_audio": {"person1": "/nvfile-heatstorage/AIGC_H100/hys/teleai_pipe/work_dirs/examples/0812/粉衣服.WAV"}
                    }
    models_path = "work_dirs/models"
    size = "720p" # ["480p", "720p"]

    ####################### 剩余为固定默认参数 ########################
    test_data["prompt"] = test_data["prompt"][:200]
    audio_mode = 0 # [0, 1] 0为标准模式, 1为音频同步强化模式
    transformer_dir = os.path.join(models_path, "transformer") 
    guidance_scale      = 1.0
    if audio_mode == 0:
        guidance_scale_audio = 1.0
    elif audio_mode == 1:
        guidance_scale_audio = 1.5
    sample_id, sample_info = sample_id, test_data
    # image_name = os.path.basename(v['ref_images'][0])[:-4]
    task = "wan-14B"
    wav2vec_dir = os.path.join(models_path, "chinese-wav2vec2-base")
    ckpt_dir = os.path.join(models_path, "Wan2.1-I2V-14B-720P")
    num_persistent_param_in_dit = None
    
    sample_shift = 2
    frame_num = 81
    motion_frame = 20
    motion_frame_ref = 1
    # input_json = v["input_json"]
    sample_text_guide_scale = guidance_scale
    sample_audio_guide_scale = guidance_scale_audio
    base_seed = 42
    sample_steps = 10
    offload_model = False
    mode = "streaming"
    lora_dir = [os.path.join(models_path, "Wan2.1_I2V_14B_4step_distill_lora_rank256.safetensors")] # models/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
    # auto sample steps switch
    if lora_dir is not None:
        if "Wan2.1_I2V_14B_4step_distill_lora_rank256.safetensors" in lora_dir[0]:
            sample_steps = 4
            sample_shift = 5
        elif "Wan2.1_I2V_14B_FusionX_LoRA.safetensors" in lora_dir[0]:
            sample_steps = 8

    # SAVE NAME
    SAVE_SUFFIX = f"_{size}" # f"_test_mt_720pmodel_720pres_multitask_1600_szr3long2_gs{guidance_scale}_apg0_step{sample_steps}_shift{sample_shift}_lora_f{frame_num}_ovlp{motion_frame}" 
    lora_scales = [1.0]
    extra_args = _parse_args()
    extra_args.size = size
    device = rank
    face_mask_area_ratio = 0

    print(f"infer {sample_info['cond_image']} // {sample_info['cond_audio']} // {sample_info['prompt']} // with device: {device}......")
    cfg = WAN_CONFIGS["ai2v-14B"]
    config_transformer = OmegaConf.load("teleai_pipe/pipelines/teleai_ai2v/config/wan_civitai.yaml")
    print(f"Generation model config: {cfg}")

    input_data = sample_info
    save_audio ="work_dirs/save_audio"
    wav2vec_feature_extractor, audio_encoder= custom_init('cpu', wav2vec_dir)
    audio_save_dir = os.path.join(save_audio, input_data['cond_image'].split('/')[-1].split('.')[0])
    os.makedirs(audio_save_dir,exist_ok=True)
    
    # human/background audio seperation
    my_clear_voice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    if len(input_data['cond_audio'])==2:
        assert False, "not supporting multi-person for now."
        new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data['cond_audio']['person1'], input_data['cond_audio']['person2'], input_data['audio_type'])
        audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
        audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
        emb1_path = os.path.join(audio_save_dir, '1.pt')
        emb2_path = os.path.join(audio_save_dir, '2.pt')
        sum_audio = os.path.join(audio_save_dir, 'sum.wav')
        sf.write(sum_audio, sum_human_speechs, 16000)
        torch.save(audio_embedding_1, emb1_path)
        torch.save(audio_embedding_2, emb2_path)
        input_data['cond_audio']['person1'] = emb1_path
        input_data['cond_audio']['person2'] = emb2_path
        input_data['video_audio'] = sum_audio
    elif len(input_data['cond_audio'])==1:
        # save full audio for saving and change speech audio for processing.
        full_audio_path = input_data['cond_audio']['person1']
        human_speech = audio_prepare_single(input_data['cond_audio']['person1'], audio_seperator=my_clear_voice)
        audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
        emb_path = os.path.join(audio_save_dir, '1.pt')
        sum_audio = os.path.join(audio_save_dir, 'sum.wav')
        sf.write(sum_audio, human_speech, 16000)
        torch.save(audio_embedding, emb_path)
        input_data['cond_audio']['person1'] = emb_path
        input_data['video_audio'] = full_audio_path

    print("Creating AI2V pipeline.")
    wan_ai2v = WanAI2VPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        transformer_dir=transformer_dir,
        config_transformer=config_transformer,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False, 
        use_usp=False,  
        t5_cpu=False,
        lora_dir=lora_dir,
        lora_scales=lora_scales,
    )

    if num_persistent_param_in_dit is not None:
        wan_ai2v.vram_management = True
        wan_ai2v.enable_vram_management(
            num_persistent_param_in_dit=num_persistent_param_in_dit
        )
    
    print("Generating video ...")
    start_time = time.time()
    video = wan_ai2v.generate(
        input_data,
        size_buckget=size,
        motion_frame=motion_frame,
        frame_num=frame_num,
        shift=sample_shift,
        sampling_steps=sample_steps,
        text_guide_scale=sample_text_guide_scale,
        audio_guide_scale=sample_audio_guide_scale,
        seed=base_seed,
        offload_model=offload_model,
        max_frames_num=frame_num if mode == 'clip' else 1000,
        face_det=None,
        face_mask_area_ratio=face_mask_area_ratio,
        color_correction_strength = extra_args.color_correction_strength,
        # motion_frame_ref=motion_frame_ref,
        extra_args=extra_args,
        )
    
    save_path = f"work_dirs/wan-videos-fun-ai2v{SAVE_SUFFIX}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if transformer_dir is not None:
        ckpt_info = transformer_dir.split("/")[-2]
    else:
        ckpt_info = "ori_model"
    print(f"audio name: {os.path.basename(input_data['cond_audio']['person1'])}")
    audio_name = os.path.basename(input_data['cond_audio']['person1'])[:-4]
    prefix = f'{ckpt_info}_{audio_name}_g{guidance_scale}_ga_{guidance_scale_audio}'
    save_file = None
    if save_file is None:
        formatted_time = datetime.now().strftime("%m%d%H%M%S")
        formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/",
                                                                    "_")[-50:]
        save_file = f"{formatted_time}_{random.randint(0, 1000)}"
    
    save_file = os.path.join(save_path, save_file)
    print(f"Saving generated video to {save_file}.mp4")
    save_video_ffmpeg(video, save_file, [input_data['video_audio']])
    # os.system(f"rm -r {save_audio}")
    print("Finished.")
    save_video_path = save_file + ".mp4"
    end_time = time.time()
    print(f"time consumed one video: {end_time - start_time}")
    return save_video_path

# 自己测试用
DEFAULT_TEST_SAMPLES = {
    "train-self-valid-1": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/chinese_squre.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-2": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/sing_169_2.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-3": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/Donald.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-4": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/banshen2.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-5": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/banshen1.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-6": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/old_man.jpg",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-7": {
        "prompt": "A person is talking, and the body movement is very natural.",
        "cond_image": "work_dirs/examples/imgs/withhand_square3.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },

    "train-self-valid-8": {
        "prompt": "A person is talking, and she walks out of the camera.",
        "cond_image": "work_dirs/examples/imgs/withhand_square2.png",
        "cond_audio": {
            "person1": "work_dirs/examples/audio/jimeng-2025-05-28-453-.wav"
        }
    },
}


# 每个进程分配一个显卡并执行任务
def run_on_gpu(rank, world_size, sample_id, test_data):
    device = torch.device(f'cuda:{rank}')  # 设置当前进程使用的GPU
    result = run_test(sample_id, test_data, device, rank)
    # print(result)

# 初始化多进程环境
def init_process(rank, world_size, DEFAULT_TEST_SAMPLES, fn, backend='nccl'):
    # 初始化分布式训练环境
    for i, (sample_id, test_data) in enumerate(DEFAULT_TEST_SAMPLES.items()):
        if i == rank:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
            fn(rank, world_size, sample_id, test_data)
    
def main():
    rank = 0
    parallel_flag = False

    # single-gpu for inference(推理时用这块即可)
    if not parallel_flag:
        run_test(sample_id=0, test_data=None, device=torch.device(f'cuda:{rank}'), rank=rank)
    # multi-gpu, only for self-test(自己多卡测试用), parallel_flag=True
    else:
        num_gpus = 8 
        world_size = num_gpus
        mp.spawn(init_process,
                    args=(world_size, DEFAULT_TEST_SAMPLES, run_on_gpu),
                    nprocs=num_gpus,  # 每个进程使用一个GPU
                    join=True)

if __name__ == '__main__':
    main()


"""
python scripts/zero_to_bf16.py  --max_shard_size 80GB --safe_serialization
"""