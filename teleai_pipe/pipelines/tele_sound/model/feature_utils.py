import os
from pathlib import Path
from typing import Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from diffusers import AutoencoderOobleck
from transformers import T5EncoderModel, T5Tokenizer

from ..ext.synchformer import Synchformer
from ..ext.autoencoder import AutoEncoderModule
from ..ext.mel_converter import get_mel_converter
from ..ext.distributions import DiagonalGaussianDistribution

class FeaturesUtils(nn.Module):

    def __init__(
        self,
        audio_vae_path: str = "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/v1-44.pth",
        vocoder_ckpt_path: str = '/nvfile-heatstorage/sgh/model/nvidia-bigvgan_v2_44khz_128band_512x',
        video_encoder_path: str = "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/synchformer_state_dict.pth",
        text_encoder_path: str = "/nvfile-heatstorage/sgh/model/t5-base",
        max_length: int = 64,
        need_vae_encoder: bool = True,
        decoder_only: bool = False
    ):
        super().__init__()
        if decoder_only:
            self.audio_vae:AutoEncoderModule = AutoEncoderModule(vae_ckpt_path=audio_vae_path,
                                                                vocoder_ckpt_path=vocoder_ckpt_path,
                                                                need_vae_encoder=False)
        else:
            self.mel_converter = get_mel_converter("44k")
            self.audio_vae:AutoEncoderModule = AutoEncoderModule(vae_ckpt_path=audio_vae_path,
                                                                vocoder_ckpt_path=vocoder_ckpt_path,
                                                                need_vae_encoder=need_vae_encoder)
            self.video_encoder:Synchformer = Synchformer()
            self.video_encoder.load_state_dict(torch.load(video_encoder_path, weights_only=True, map_location='cpu'))
            self.video_encoder.to(dtype=torch.float16)
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(text_encoder_path, max_length=max_length, legacy=True)
            self.text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(text_encoder_path)
            self.max_length:int = max_length

    def compile(self):
        if self.audio_vae is not None:
            self.audio_vae = torch.compile(self.audio_vae)
        if self.video_encoder is not None:
            self.video_encoder = torch.compile(self.video_encoder)
        if self.text_encoder:
            self.text_encoder = torch.compile(self.text_encoder)

    @torch.inference_mode()
    def encode_video(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.video_encoder is not None, 'video_encoder is not loaded'
        # x: (B, T, C, H, W) : 12, 25*8, 3, 224, 224
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1 # (200 - 16) // 8 + 1 = 24
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W): 12, 24, 16, 3, 224, 224

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.video_encoder(x[i:i + batch_size])) # input: 12, 1, 16, 3, 224, 224
        x = torch.cat(outputs, dim=0)

        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b) # 12, 24*8, 768

        return x

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        assert self.text_encoder is not None, 'T5 is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state

    @torch.inference_mode()
    def encode_audio(self, x) -> torch.Tensor:
        assert self.audio_vae is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        audio_laten = self.audio_vae.encode(mel)
        return audio_laten.sample().transpose(1, 2)

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.audio_vae is not None, 'VAE is not loaded'
        return self.audio_vae.vocode(mel)
    
    @torch.inference_mode()
    def decode_audio(self, z: torch.Tensor) -> torch.Tensor:
        assert self.audio_vae is not None, 'VAE is not loaded'
        return self.audio_vae.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
