import ipdb
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from loguru import logger

device = 'cuda:0'

def _load_file(file_path, dtype=None):
    with safe_open(file_path, framework="pt", device=device) as f:
        tensor_dict = {key: f.get_tensor(key).to(dtype) if dtype is not None else f.get_tensor(key) for key in f.keys()}
    return tensor_dict

def _apply_lora_weights(weight_dict, lora_weights, alpha):
    lora_pairs = {}
    lora_diffs = {}

    def try_lora_pair(key, suffix_a, suffix_b, target_suffix, prefix):
        if key.endswith(suffix_a):
            base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
            pair_key = key.replace(suffix_a, suffix_b)
            if pair_key in lora_weights:
                lora_pairs[base_name] = (key, pair_key)

    def try_lora_diff(key, suffix, target_suffix, prefix):
        if key.endswith(suffix):
            base_name = key[len(prefix) :].replace(suffix, target_suffix)
            lora_diffs[base_name] = key

    for prefix in ["", "diffusion_model."]:
        for key in lora_weights.keys():
            if not key.startswith(prefix):
                continue

            try_lora_pair(key, "lora_A.weight", "lora_B.weight", "weight", prefix)
            try_lora_pair(
                key, "lora_down.weight", "lora_up.weight", "weight", prefix
            )
            try_lora_diff(key, "diff", "weight", prefix)
            try_lora_diff(key, "diff_b", "bias", prefix)
            try_lora_diff(key, "diff_m", "modulation", prefix)

    applied_count = 0
    for name, param in weight_dict.items():
        if name in lora_pairs:
            name_lora_A, name_lora_B = lora_pairs[name]
            param = param.to(torch.bfloat16)
            lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
            lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
            scaling_factor = 1.0
            if param.shape == (lora_B.shape[0], lora_A.shape[1]):
                lora_alpha = lora_weights.get(
                    f'diffusion_model.{name.replace(".weight", ".alpha")}', None
                )
                if lora_alpha is not None:
                    rank = lora_A.shape[0]
                    scaling_factor = float(lora_alpha) / rank
                else:
                    raise RuntimeError("should not happen")
                param += torch.matmul(lora_B, lora_A) * alpha * scaling_factor
                param = param.to(torch.float32)
                weight_dict[name] = param
                # if name == 'blocks.0.self_attn.q.weight':
                #     print("DEBUG SHOW:")
                #     print(f"{alpha}  {scaling_factor}")
                #     print(f"{lora_B.dtype} {lora_A.dtype} ")
                #     print(torch.matmul(lora_B, lora_A) * alpha * scaling_factor)
                applied_count += 1
            else:
                raise RuntimeError("should not happen again.")
        elif name in lora_diffs:
            name_diff = lora_diffs[name]
            lora_diff = lora_weights[name_diff].to(param.device, param.dtype)
            if param.shape == lora_diff.shape:
                param += lora_diff * alpha
                applied_count += 1

    logger.info(f"Applied {applied_count} LoRA weight adjustments")
    if applied_count == 0:
        logger.info(
            "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
        )

if __name__ == '__main__':
    source_low_dit_dir = Path('/root/models/Wan2.2-I2V-A14B/low_noise_model_bkp/')
    target_low_dit_dir = Path('/root/models/Wan2.2-I2V-A14B/low_noise_model_fused_lora/')
    lora_dit_low_dir = Path('/root/models/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors')

    # load lora
    loras = _load_file(lora_dit_low_dir, dtype=torch.bfloat16)
    # loras = _load_file(lora_dit_low_dir)

    tensor_splits = 6
    for ind in range(1, tensor_splits+1):
        low_tensors = _load_file(source_low_dit_dir / f'diffusion_pytorch_model-0000{ind}-of-0000{tensor_splits}.safetensors')
        _apply_lora_weights(low_tensors, loras, 1.0)
        if not target_low_dit_dir.is_dir():
            target_low_dit_dir.mkdir(exist_ok=True)
        save_file(low_tensors, target_low_dit_dir / f'diffusion_pytorch_model-0000{ind}-of-0000{tensor_splits}.safetensors')

        # print(low_tensors.keys())
        # ipdb.set_trace()
