import torch
from torch import nn
from teleai_pipe.models.teleai_t2s_masked_transformer import MaskTransformer
from teleai_pipe.models.teleai_t2s_vqcvae import VQCVAE


class Text2MotionTransformer(nn.Module):
    def __init__(self, transformer_opt: dict, vqcvae_opt: dict) -> None:
        super(Text2MotionTransformer, self).__init__()
        self.transformer_opt = transformer_opt
        self.vqcvae_opt = vqcvae_opt

        # build modules
        self.transformer = MaskTransformer(**transformer_opt)

        # 加载vqvae
        self.vqvae = VQCVAE(**vqcvae_opt)

    def generate(self, batch_lens, batch_control, batch_caption, infer_opt):
        batch_lens = torch.tensor(batch_lens, dtype=torch.int64).to(
            device=self.transformer.device
        )
        batch_code_lens = self.vqvae.get_code_length(batch_lens)
        batch_control = self.vqvae.control_input_process(
            [c.to(device=self.transformer.device) for c in batch_control]
        )
        batch_ids = self.transformer.generate(
            batch_caption,
            batch_code_lens,
            timesteps=infer_opt["timesteps"],
            cond_scale=infer_opt["cond_scale"],
            temperature=infer_opt["temperature"],
            topk_filter_thres=infer_opt["topk_filter_thres"],
            gsample=infer_opt["gsample"],
            force_mask=infer_opt["force_mask"],
        )
        batch_code_windows, batch_info = self.vqvae.pack_code_windows(
            batch_ids, batch_lens
        )
        batch_motion = self.vqvae.batch_chain_decode(
            batch_code_windows, batch_control, batch_info
        )
        generations = self.vqvae.output_process(batch_motion, batch_lens)
        return generations

    def normally_generate(self, batch_lens, batch_caption, infer_opt):
        batch_lens = torch.tensor(batch_lens, dtype=torch.int64).to(
            device=self.transformer.device
        )
        batch_code_lens = self.vqvae.get_code_length(batch_lens)
        batch_ids = self.transformer.generate(
            batch_caption,
            batch_code_lens,
            timesteps=infer_opt["timesteps"],
            cond_scale=infer_opt["cond_scale"],
            temperature=infer_opt["temperature"],
            topk_filter_thres=infer_opt["topk_filter_thres"],
            gsample=infer_opt["gsample"],
            force_mask=infer_opt["force_mask"],
        )
        batch_code_windows, batch_info = self.vqvae.pack_code_windows(
            batch_ids, batch_lens
        )
        batch_motion = self.vqvae.batch_decode(batch_code_windows, batch_info)
        generations = self.vqvae.output_process(batch_motion, batch_lens)
        return generations

    def eidt(self, controls, motions):
        controls = [c.to(device=self.transformer.device) for c in controls]
        motions = [m.to(device=self.transformer.device) for m in motions]
        batch_motion, m_length = self.vqvae.input_process(motions)
        batch_control = self.vqvae.control_input_process(controls)

        batch_code_windows, batch_info = self.vqvae.batch_encode(batch_motion, m_length)
        batch_recon = self.vqvae.batch_chain_decode(
            batch_code_windows, batch_control, batch_info
        )
        recon2 = self.vqvae.output_process(batch_recon, m_length)
        return recon2

    def concat(self, motion1, motion2):
        pass
