import logging
from dataclasses import dataclass
from argparse import ArgumentParser
from pathlib import Path
import math
from typing import Tuple, Optional, Union, Literal

import random
import numpy as np
import sys, os

import torch
import torchaudio
from torchaudio.functional import resample
from torch.nn.functional import cosine_similarity
from torchvision.transforms import v2

from einops import rearrange

from .tele_sound.scheduler.flow_matching import FlowMatching
from .tele_sound.model.model import get_model
from .tele_sound.model.feature_utils import FeaturesUtils

from .tele_sound.ext.imagebind.models import imagebind_model
from .tele_sound.ext.imagebind.models.imagebind_model import ModalityType

from .tele_sound.ext.imagebind.models.multimodal_preprocessors import SimpleTokenizer

from .tele_sound.ext.vidMuse.VidMuse_app import _do_predictions
from .tele_sound.ext.vidMuse.audiocraft.models import VidMuse

@dataclass
class SequenceConfig:
    # general
    duration: float

    # audio
    sampling_rate: int
    spectrogram_frame_rate: int
    latent_downsample_rate: int = 2

    # visual
    sync_frame_rate: int = 25
    sync_num_frames_per_segment: int = 16
    sync_step_size: int = 8
    sync_downsample_rate: int = 2

    @property
    def num_audio_frames(self) -> int:
        # we need an integer number of latents
        return self.latent_seq_len * self.spectrogram_frame_rate * self.latent_downsample_rate

    @property
    def latent_seq_len(self) -> int:
        return int(
            math.ceil(self.duration * self.sampling_rate / self.spectrogram_frame_rate /
                      self.latent_downsample_rate))

    @property
    def sync_seq_len(self) -> int:
        num_frames = self.duration * self.sync_frame_rate
        num_segments = (num_frames - self.sync_num_frames_per_segment) // self.sync_step_size + 1
        return int(num_segments * self.sync_num_frames_per_segment / self.sync_downsample_rate)



def load_and_transform_text(text, tokenizer):
    if text is None:
        return None
    tokens = [tokenizer(t).unsqueeze(0) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10
def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank

def process_data(videos_tensor, audios_tensor, duration, clip_duation=2, fps=25, sr=44100, target_sr=16000):
    audios_tensor = audios_tensor[:,:sr*duration]
    num_clips = math.ceil(duration / clip_duation)
    duration_step_size = (duration - clip_duation) / (num_clips - 1)
    duration_index = [i * duration_step_size for i in range(num_clips)]
    if target_sr != sr:
        audios_tensor = torchaudio.functional.resample(
            audios_tensor, orig_freq=sr, new_freq=target_sr
        )
    videos_list = []
    audios_list = []
    for idx in duration_index:
        clip_video = videos_tensor[:,[int(idx*fps), int((idx+clip_duation/2)*fps)],...]
        clip_audio = audios_tensor[...,int(idx*target_sr):int((idx+clip_duation)*target_sr)]
        clip_mels = []
        for b in range(clip_audio.shape[0]):
            audio = clip_audio[b]
            mel = waveform2melspec(
                audio, sample_rate=target_sr, num_mel_bins=128, target_length=204
            )
            clip_mels.append(mel)
        clip_mels = torch.stack(clip_mels, dim=0)
        videos_list.append(clip_video)
        audios_list.append(clip_mels)

    videos_clips = torch.stack(videos_list, dim=1).permute(0, 1, 3, 2, 4, 5)
    audios_clips = torch.stack(audios_list, dim=1)
    return videos_clips, audios_clips

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class TelesSoundPipeline(torch.nn.Module):
    
    def __init__(self, 
                model_config,
                num_steps=25,
                num_samples=16,
                related_threhold=0.2,
                device="cuda:0",
                dtype=torch.bfloat16,
                seed=42):
        super().__init__()
        set_seed(seed)
        self.model_config = model_config
        self.num_samples = num_samples
        self.related_threhold = related_threhold
        self.model = get_model("large").to(device, dtype).eval()
        self.model.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))
        self.scheduler = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)
        
        self.feature_utils = FeaturesUtils(
            audio_vae_path=model_config.audio_vae_path,
            vocoder_ckpt_path=model_config.vocoder_ckpt_path,
            video_encoder_path=model_config.video_encoder_path,
            text_encoder_path=model_config.text_encoder_path,
            max_length=model_config.text_max_length,
            need_vae_encoder=False,
            decoder_only=False
        )
        self.feature_utils = self.feature_utils.to(device, dtype).eval()
        self.tokenizer = SimpleTokenizer(bpe_path=model_config.bpe_path)
        self.related_model = imagebind_model.imagebind_huge(pretrained_path=model_config.imagebind_path).to(device,dtype).eval()

        self.vidmuse = VidMuse.get_pretrained(model_config.vidmuse_ckpt_path, device)

    def read_video(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        duration: Optional[float] = None,
        crop_size: Optional[Tuple[int, int]] = (224, 224),
     ) -> torch.Tensor:
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)
        if duration is not None:
            real_duration = int(min(total_frames / original_fps, duration))
        else:
            real_duration = int(total_frames / original_fps)
        target_times = np.linspace(
            0, real_duration, int(target_fps * real_duration), endpoint=False
        )
        target_indices = np.round(target_times * original_fps).astype(int)
        frames = vr.get_batch(target_indices)

        video = frames.permute(0, 3, 1, 2)
        video_transform = v2.Compose(
            [
                v2.Resize(crop_size, interpolation=v2.InterpolationMode.BICUBIC),
                # v2.CenterCrop(crop_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        video = video_transform(video)
        return video, real_duration

    def VidMuse(self, video_path, topk=250, topp=0.0, temperature=1.0, cfg_coef=3.0):
        print('................use VidMuse ................')
          
        #os.chdir(self.vidmuse_path)
        # # 设置 Hugging Face 缓存相对路径（相对于工作目录）
        # os.environ["HF_HOME"] = "./model/huggingface"
        # os.environ["TRANSFORMERS_CACHE"] = "./model/huggingface"
        # os.environ["HF_DATASETS_CACHE"] = "./model/huggingface"

        wavs = _do_predictions(self.vidmuse,
            [str(video_path)], duration=30, progress=False,
            top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef
        )
        
        wavs = resample(wavs, self.vidmuse.sample_rate, 44100)

        return wavs


    @torch.no_grad()
    def __call__(
        self,
        video_path,
        # duration,
        prompt=None,
        negative_prompt=None,
        cfg_strength=4.5,
        num_inference_steps=25,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        self.scheduler.num_steps = num_inference_steps
        video, duration = self.read_video(str(video_path), target_fps=25)
        origin_batch_videos = video.unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
        batch_videos = self.feature_utils.encode_video(origin_batch_videos.to(device, dtype)).to(dtype).detach().clone()
        SEQCONFIG = SequenceConfig(duration=duration, sampling_rate=44100, spectrogram_frame_rate=512)
        self.model.update_seq_lengths(SEQCONFIG.latent_seq_len, SEQCONFIG.sync_seq_len)
        
        x0 = torch.randn(self.num_samples,
                SEQCONFIG.latent_seq_len,
                40,
                device=device,
                dtype=dtype)
       
        if prompt is None:
            prompts = ["" for _ in range(self.num_samples)]
        else:
            prompts = [prompt for _ in range(self.num_samples)]

        text_feature = self.feature_utils.encode_text(prompts).to(dtype).detach().clone()
        
        if negative_prompt is None:
            negative_text_feature = None
        else:
            negative_prompts = [negative_prompt for _ in range(self.num_samples)]
            negative_text_feature = self.feature_utils.encode_text(negative_prompts).to(dtype).detach().clone()

        cfg_ode_wrapper = lambda t, x: self.model.ode_wrapper(t, x, batch_videos, text_feature, cfg_strength, negative_text_feature=negative_text_feature)
        x1_hat = self.scheduler.to_data(cfg_ode_wrapper, x0)
        x1_hat = self.model.unnormalize(x1_hat)
        mel = self.feature_utils.decode_audio(x1_hat)
        audios = self.feature_utils.vocode(mel).float().cpu()

        # #dedbug
        # for i in range(audios.shape[0]):
        #     torchaudio.save('/gemini/space/szhang/video2sound/demo_test' + f"{i}.flac", audios[i,:], 44100)

        videos_tensor, audios_tensor= process_data(origin_batch_videos.cpu(), audios.cpu(), duration)

        text_tokens = load_and_transform_text(prompts, self.tokenizer)
        with torch.no_grad():
            embeddings = self.related_model({
                ModalityType.AUDIO: audios_tensor.to(device,dtype),
                ModalityType.VISION: videos_tensor.to(device,dtype),
                ModalityType.TEXT: text_tokens.to(device),
            })

        sim_va = cosine_similarity(
            embeddings[ModalityType.AUDIO], 
            embeddings[ModalityType.VISION]
        ).cpu()

        max_sim_va = torch.max(sim_va)
        max_sim_va_idx = torch.argmax(sim_va)

        sim_ta = cosine_similarity(
            embeddings[ModalityType.AUDIO], 
            embeddings[ModalityType.TEXT]
        ).cpu()

        max_sim_ta = torch.max(sim_ta)
        #empty_audio = torch.zeros((audios.shape[1], audios.shape[2],), dtype=audios.dtype)
        print(f'max_sim_va: {max_sim_va}, max_sim_ta: {max_sim_ta}')
        
        if max_sim_va < 0.2 and max_sim_ta < 0.2:
            #return empty_audio            
            return self.VidMuse(video_path)

        if max_sim_va > 0.3:
            return audios[max_sim_va_idx]
        
        mask_1 = (sim_va > 0.2)
        mask_2 = (sim_ta > 0.2)
        mask = mask_1 | mask_2

        filter_audios = audios[mask]
        filter_sim_va = sim_va[mask]
        filter_sim_ta = sim_ta[mask]
        if filter_audios is not None:
            weights = 0.3 * filter_sim_va + 0.7 * filter_sim_ta
            idx = torch.argmax(weights)
            return filter_audios[idx]

        #return empty_audio
        return self.VidMuse(video_path)
