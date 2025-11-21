from typing import Any, Mapping
import sys
import torch
from torch.nn import functional as F

sys.path.insert(0, '.')  # nopep8
from .modules.feat_extractors.audio.ast import AST

def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Synchformer(torch.nn.Module):
    ''' The module has similar structure to SparseSync (SparseSync) but has a diffrerent
    forward pass. It expects the output of the feature extractors to have global and
    segment-level representations.'''

    def __init__(self, audio_sync_model_path=None):
        super().__init__()
        self.afeat_extractor = AST(
            ckpt_path=audio_sync_model_path,
            extract_features=True,
            max_spec_t=66,
            factorize_freq_time=True,
            agg_freq_module="TransformerEncoderLayer",
            agg_time_module="torch.nn.Identity",
            add_global_repr=False
        )
    
    def extract_synch_afeats(self, aud: torch.Tensor, for_loop=False, aud_mask: torch.Tensor = None):
        aud = self.extract_afeats(aud, for_loop, aud_mask=aud_mask)
        aud = self.aproj(aud)
        # flatten the segment dim (treating the sequence of segments as a single sequence)
        B, S, ta, D = aud.shape
        aud = aud.view(B, S*ta, D)  # (B, S*ta, D)
        return aud

    def extract_afeats(self, aud, for_loop, aud_mask=None):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        if aud_mask is not None:
            aud_mask = aud_mask.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud, for_loop=for_loop, cont_mask=aud_mask)
        return aud

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        ''' discard all entries except afeat_extractor.'''
        sd_ = dict()
        for k, v in sd.items():
            if k.startswith('afeat_extractor'):
                sd_[k]=v
            elif k.startswith('a_encoder'):
                sd_[k.replace("a_encoder", 'afeat_extractor')] = v  
        # sd = {k: v for k, v in sd.items() if k.startswith('afeat_extractor')}
        return super().load_state_dict(sd_, strict)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from time import time

    device = torch.device('cpu')

    model = Synchformer()
    ckpt = torch.load("/gemini-1/space/human_guozz2/songqy/vast_old/pretrained/synchformer/24-01-04T16-39-21.pt", weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    start_time = time()
    for i in range(3):
        # vis = torch.rand(1, 125, 3, 224, 224, device=device)
        aud = torch.rand(5, 8, 1, 128, 66, device=device)
        # cls_logits, off_logits, sync_logits = model(vis, aud)
        # inference in half precision
        out = model.extract_afeats(aud, for_loop=False)
    print('Time:', round(time() - start_time, 3))
