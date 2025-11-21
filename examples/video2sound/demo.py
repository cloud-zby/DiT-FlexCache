from typing import Tuple, Optional

import torch
import torchaudio
from torchvision.transforms import v2

import subprocess
import numpy as np
import os
import random
from pathlib import Path
from teleai_pipe.pipelines.telesound_pipeline import TelesSoundPipeline
from dataclasses import dataclass


@dataclass
class Config:
    model_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/sound_effect_with_blank_voice_vgg_90000.pth"
    audio_vae_path: str = "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/v1-44.pth"
    vocoder_ckpt_path: str = (
        "/nvfile-heatstorage/sgh/model/nvidia-bigvgan_v2_44khz_128band_512x"
    )
    video_encoder_path: str = (
        "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/synchformer_state_dict.pth"
    )
    text_encoder_path: str = "/nvfile-heatstorage/sgh/model/t5-base"
    text_max_length: int = 64
    imagebind_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/imagebind_huge.pth"
    bpe_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/bpe_simple_vocab_16e6.txt.gz"
    vidmuse_ckpt_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/music"
    # encodec_32khz_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/music/facebook__encodec_32khz"
    # clip_vit_base_patch32: str = "/nvfile-heatstorage/sgh/share/tele_sound/music/vit"

# init pipeline
model_config = Config()
pipeline = TelesSoundPipeline(
    model_config=model_config, num_steps=25, num_samples=16, related_threhold=0.2, device="cuda", dtype=torch.bfloat16, seed=114514
)
def telesound_demo(prompt, input_video_file, output_dir):

    # if not os.path.exists(input_video_file):
    #     print(f"{input_video_file} not found")
    #     os._exit(0)

    if not input_video_file.exists():
        print(f"{input_video_file} not found")
        os._exit(0)
    os.makedirs(output_dir, exist_ok=True)

    #video, duration = read_video(str(input_video_file), target_fps=25)

    # generate
    audio = pipeline(
        video_path=input_video_file,
        prompt=prompt,
        cfg_strength=4.5,
        num_inference_steps=25,
    )


    torchaudio.save(output_dir + f"{input_video_file.stem}.flac", audio, 44100)
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", input_video_file, "-i", f"{output_dir + f'{input_video_file.stem}.flac'}", "-c:v", "copy", "-c:a", "libmp3lame", "-map", "0:v:0", "-map", "1:a:0", "-shortest", f"{output_dir + f'{input_video_file.stem}.mp4'}"])

if __name__ == "__main__":
    prompts = ['knife chopping', 'none', "none"]
    input_video_files = [Path("/nvfile-heatstorage/mah/share/一位中年男子正在厨房里专注地切割食材_他穿着简单的.mp4"), Path('/nvfile-heatstorage/mah/share/写实风格，云裳跃入空中，汉服如同一幅展开的画卷，背景是西湖的广阔水面，阳光下她的身形如同飞翔的鸟，线条优美，动作流畅。.mp4'), Path('/nvfile-heatstorage/mah/share/舞蹈动作5.mp4')]
    output_dir = "./new_demo/"
    for prompt, input_video_file in zip(prompts, input_video_files):
        telesound_demo(prompt, input_video_file, output_dir)
    
