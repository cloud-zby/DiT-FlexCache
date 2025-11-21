from cryptography.fernet import Fernet
import torch
import io, os
from tqdm import tqdm
from safetensors.torch import load_file
from typing import List, Optional, Union
import safetensors

# ========= diffusers ========= #
from diffusers.utils import (
    GGUF_FILE_EXTENSION,
    SAFETENSORS_FILE_EXTENSION,
    is_torch_version,
)
from diffusers.models.model_loading_utils import load_gguf_checkpoint

# ============ key ============ #
key = b'LxfQgPiUiXy_fxVUlLlpusLdxdEMQM_ltzY_28kXerM='   # 使用key = Fernet.generate_key()随机生成


def encryption_model(origin_file, encrypted_file):
    model = load_file(origin_file)
    b = io.BytesIO()

    torch.save(model, b)
    b.seek(0)
    pth_bytes = b.read()
    encrypted_data = Fernet(key).encrypt(pth_bytes)
    with open(encrypted_file, 'wb') as fw:
        fw.write(encrypted_data)
        
        
def decryption_model(encrypt_weight_path):
    with open(encrypt_weight_path, 'rb') as fr:
        encrypted_data = fr.read()
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    checkpoint = torch.load(b, map_location='cpu')

    return checkpoint


def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    # TODO: We merge the sharded checkpoints in case we're doing quantization. We can revisit this change
    # when refactoring the _merge_sharded_checkpoints() method later.
    if isinstance(checkpoint_file, dict):
        return checkpoint_file
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            if os.path.exists(checkpoint_file.replace('.dll', '')):
                return safetensors.torch.load_file(checkpoint_file, device="cpu")
            else:
                return decryption_model(checkpoint_file)
        elif file_extension == GGUF_FILE_EXTENSION:
            return load_gguf_checkpoint(checkpoint_file)
        else:
            weights_only_kwarg = {"weights_only": True} if is_torch_version(">=", "1.13") else {}
            return torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )
        

if __name__ == '__main__':
    # 需加密模型列表
    model_list = [
        '/nvfile-heatstorage/cv_banc/zangxh/projects/teleai_pipe/teleai_pipe/pipelines/teleai_edit_v2/models/lora/checkpoint-35000/pytorch_lora_weights.safetensors',
    ]
    
    for model_name in tqdm(model_list):
        encryption_model(model_name, model_name.replace('.safetensors', '.dll.safetensors'))
