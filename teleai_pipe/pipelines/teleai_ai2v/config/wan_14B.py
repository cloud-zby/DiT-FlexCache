# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V 14B ------------------------#

wan_14B_config = EasyDict(__name__='Config: Wan AI2V 14B')
wan_14B_config.update(wan_shared_cfg)
wan_14B_config.sample_neg_prompt = 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'

wan_14B_config.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
wan_14B_config.t5_tokenizer = 'google/umt5-xxl'

# clip
wan_14B_config.clip_model = 'clip_xlm_roberta_vit_h_14'
wan_14B_config.clip_dtype = torch.float16
wan_14B_config.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
wan_14B_config.clip_tokenizer = 'xlm-roberta-large'

# vae
wan_14B_config.vae_checkpoint = 'Wan2.1_VAE.pth'
wan_14B_config.vae_stride = (4, 8, 8)

# transformer
wan_14B_config.patch_size = (1, 2, 2)
wan_14B_config.dim = 5120
wan_14B_config.ffn_dim = 13824
wan_14B_config.freq_dim = 256
wan_14B_config.num_heads = 40
wan_14B_config.num_layers = 40
wan_14B_config.window_size = (-1, -1)
wan_14B_config.qk_norm = True
wan_14B_config.cross_attn_norm = True
wan_14B_config.eps = 1e-6
