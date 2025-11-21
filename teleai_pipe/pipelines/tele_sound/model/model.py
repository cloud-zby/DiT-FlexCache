import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.embeddings import TimestepEmbedder, compute_rope_rotations
from .modules.nets import MLP, ChannelLastConv1d, ConvMLP
from .modules.layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()

class AudioDiffusion(nn.Module):
    def __init__(self,
                 *,
                 audio_latent_dim: int,
                 video_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 video_seq_len: int,
                 text_seq_len: int = 64,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.audio_latent_dim = audio_latent_dim
        self._latent_seq_len = latent_seq_len
        self._video_seq_len = video_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(audio_latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        self.video_input_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.video_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # self.video_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, video_dim)))
        self.t_embed = TimestepEmbedder(hidden_dim,
                                        frequency_embedding_size=hidden_dim,
                                        max_period=1)

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(audio_latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(audio_latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == audio_latent_dim, f'{latent_mean.numel()=} != {audio_latent_dim=}'
        
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for _ in range(fused_depth)
        ])
        
        self.final_layer = FinalBlock(hidden_dim, audio_latent_dim)
        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        video_rot = compute_rope_rotations(self._video_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * 1.5,
                                          device=self.device)
        text_rot = compute_rope_rotations(self._text_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * 2,
                                          device=self.device)
        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.video_rot = nn.Buffer(video_rot, persistent=False)
        self.text_rot = nn.Buffer(text_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, _video_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._video_seq_len = _video_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.audio_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.audio_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.video_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.video_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def forward(self, audio_latent: torch.Tensor, video: torch.Tensor, text: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, N, C) 
        vf: (B, T, C_V)
        t: (B,)
        """
        assert audio_latent.shape[1] == self._latent_seq_len, f'{audio_latent.shape=} {self._latent_seq_len=}'
        assert video.shape[1] == self._video_seq_len, f'{video.shape=} {self._video_seq_len=}'
        assert text.shape[1] == self._text_seq_len, f'{text.shape=} {self._text_seq_len=}'

        audio_latent = self.audio_input_proj(audio_latent)
        
        video = self.video_input_proj(video)  # (B, VN, D)
        text = self.text_input_proj(text)  # (B, VN, D)
        video_c = video.transpose(1, 2).clone()  # (B, D, VN)
        video_c:torch.Tensor = F.interpolate(video_c, size=self._latent_seq_len, mode='nearest-exact')
        video_c = video_c.transpose(1, 2)  # (B, N, D)
        video_c = self.video_cond_proj(video_c.mean(dim=1))  # (B, D)
        text_c = self.text_cond_proj(text.mean(dim=1))  # (B, D)
        global_condition = self.global_cond_mlp(video_c + text_c)
        global_condition = self.t_embed(t).unsqueeze(1) + global_condition.unsqueeze(1)

        for block in self.joint_blocks:
            audio_latent, video, text = block(audio_latent, video, text, global_condition, self.latent_rot, self.video_rot, self.text_rot)  # (B, N, D)

        for block in self.fused_blocks:
            audio_latent = block(audio_latent, global_condition, self.latent_rot)

        flow = self.final_layer(audio_latent, global_condition)
        return flow

    def ode_wrapper(self, t: torch.Tensor, audio_latent: torch.Tensor, video: torch.Tensor, text: torch.Tensor, cfg_strength: float, *, negative_text_feature: torch.Tensor = None) -> torch.Tensor:
        t = t * torch.ones(len(audio_latent), device=audio_latent.device, dtype=audio_latent.dtype)

        if cfg_strength < 1.0:
            return self.forward(audio_latent, video, text, t)
        else:
            if negative_text_feature is None:
                return (cfg_strength * self.forward(audio_latent, video, text, t) +
                    (1 - cfg_strength) * self.forward(audio_latent, torch.zeros_like(video), torch.zeros_like(text), t))
            else:
                return (cfg_strength * self.forward(audio_latent, video, text, t) +
                    (1 - cfg_strength) * self.forward(audio_latent, torch.zeros_like(video), negative_text_feature, t))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'video_rot' in src_dict:
            del src_dict['video_rot']
        if 'text_rot' in src_dict:
            del src_dict['text_rot']

        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def video_seq_len(self) -> int:
        return self._video_seq_len

    @property
    def text_seq_len(self) -> int:
        return self._text_seq_len
        
def get_model(name:str = "large", **kwargs) -> AudioDiffusion:
    if name == "large":
        num_heads = 16
        return AudioDiffusion(audio_latent_dim=40,
                    video_dim=768,
                    text_dim=768,
                    hidden_dim=64 * num_heads,
                    depth=36,
                    fused_depth=24,
                    num_heads=num_heads,
                    latent_seq_len=345,
                    video_seq_len=192,
                    text_seq_len=64,
                    **kwargs)
    elif name == "soundeffect":
        num_heads = 16
        return AudioDiffusion(audio_latent_dim=40,
                    video_dim=768,
                    text_dim=768,
                    hidden_dim=64 * num_heads,
                    depth=36,
                    fused_depth=24,
                    num_heads=num_heads,
                    latent_seq_len=690,
                    video_seq_len=392,
                    text_seq_len=64,
                    **kwargs)
    else:
        num_heads = 7
        return AudioDiffusion(audio_latent_dim=40,
                    video_dim=768,
                    text_dim=768,
                    hidden_dim=64 * num_heads,
                    depth=12,
                    fused_depth=8,
                    num_heads=num_heads,
                    latent_seq_len=345,
                    video_seq_len=192,
                    text_seq_len=64,
                    **kwargs)
