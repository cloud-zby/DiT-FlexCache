import torch
from torch import nn
from loguru import logger
from torch.utils.checkpoint import checkpoint
from transformers import pipeline
# from sbp.nn.model_paths import MODEL_PATHS


class ImageEncoderWithEVA(nn.Module):

    def __init__(self, output_dim, base_model='eva02_large_patch14_448.mim_in22k_ft_in1k', layer_num=6, seq_len=3, device='cpu', use_pe=False):
        super().__init__()
        self.output_dim = output_dim
        import timm
        # paths = {
        #     'eva02_large_patch14_448.mim_in22k_ft_in1k': MODEL_PATHS.EVA02_LARGE_448_MIM_IN22K,
        #     'eva02_base_patch14_224.mim_in22k': MODEL_PATHS.EVA02_BASE_224_MIM_IN22K,
        # }
        if base_model == 'eva02_base_patch14_224.mim_in22k':
            self.img_seq = 257
        elif base_model == 'eva02_large_patch14_448.mim_in22k_ft_in1k':
            self.img_seq = 1025
        elif base_model == 'siglip2':
            self.img_seq = 1024
        else:
            raise ValueError(f" unknown {base_model}")
        # base_model = timm.create_model(base_model, pretrained=True, pretrained_cfg_overlay={'file': paths[base_model], 'custom_load': False})
        base_model = timm.create_model(base_model, pretrained=False)
        del base_model.norm, base_model.fc_norm, base_model.head, base_model.head_drop
        del base_model.blocks[layer_num:]
        self.base_model = base_model
        # self.base_model.cls_token.to(device=device, dtype=torch.bfloat16)
        # self.base_model.reg_token.to(device=device, dtype=torch.bfloat16)
        # self.base_model.pos_embed.to(device=device, dtype=torch.bfloat16)
        self.project = nn.Linear(base_model.num_features, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        self.seq_len = seq_len
        self.device = device
        self.use_pe = use_pe
        if self.use_pe:
            self.pe = torch.zeros([1, self.seq_len * self.img_seq, self.output_dim], device=self.device, dtype=torch.bfloat16)
            for i in range(self.seq_len):
                self.pe[:, i * self.img_seq: (i + 1) * self.img_seq, i::self.seq_len] = 0.05

    def forward(self, image_list):
        splits = [len(lst) for lst in image_list]
        if sum(splits) == 0:
            return torch.zeros([len(splits), self.seq_len * self.img_seq, self.output_dim], dtype=torch.bfloat16)
        x = torch.concat(image_list, dim=0).to(device=self.device, dtype=torch.bfloat16)
        x = self.base_model.patch_embed(x)
        x, rot_pos_embed = self.base_model._pos_embed(x)
        for blk in self.base_model.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.project(x)
        x = self.final_norm(x)
        b, seq_len, c= x.shape
        split_patches = torch.split(x, splits, dim=0)
        split_patches = [nn.functional.pad(sample, (0, 0, 0, 0, 0, self.seq_len - len(sample))) for sample in split_patches]
        x = torch.stack(split_patches, dim=0)
        x = x.reshape((len(splits), self.seq_len * seq_len, c))
        if self.use_pe:
            x = x + self.pe
        return x

    @staticmethod
    def state_dict_converter():
        return ImageEncoderConverter()


class ImageEncoderConverter:

    def from_huggingface(self, state_dict):
        layer_num = 12 if any(['blocks.10' in k for k in state_dict]) else 9
        return state_dict, {'output_dim': 4096, "layer_num": layer_num, "seq_len": 2, "device": 'cuda:0'} 

    def from_civitai(self, state_dict):
        layer_num = 12 if any(['blocks.10' in k for k in state_dict]) else 9
        print("init layer_num:", layer_num)
        return state_dict, {'output_dim': 4096, "layer_num": layer_num, "seq_len": 2, "device": 'cuda:0'}   