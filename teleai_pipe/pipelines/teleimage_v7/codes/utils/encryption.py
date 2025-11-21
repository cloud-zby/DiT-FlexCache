from cryptography.fernet import Fernet
import torch
import io
from tqdm import tqdm
from safetensors.torch import load_file
import json
from safetensors import safe_open
key = b'LxfQgPiUiXy_fxVUlLlpusLdxdEMQM_ltzY_28kXerM='   # 使用key = Fernet.generate_key()随机生成


# def encryption_model(origin_file, encrypted_file):
#     model = load_file(origin_file)
#     b = io.BytesIO()

#     torch.save(model, b)
#     b.seek(0)
#     pth_bytes = b.read()
#     encrypted_data = Fernet(key).encrypt(pth_bytes)
#     with open(encrypted_file, 'wb') as fw:
#         fw.write(encrypted_data)
        
        
# def decryption_model(encrypt_weight_path, key=b'LxfQgPiUiXy_fxVUlLlpusLdxdEMQM_ltzY_28kXerM='):
#     with open(encrypt_weight_path, 'rb') as fr:
#         encrypted_data = fr.read()
#     decrypted_data = Fernet(key).decrypt(encrypted_data)
#     b = io.BytesIO(decrypted_data)
#     b.seek(0)
#     checkpoint = torch.load(b, map_location='cpu')

#     return checkpoint


def _load_sft_state_dict_metadata(model_file: str):
    """原版的safetensor读取metadata的方法（保持不变）"""
    import safetensors.torch
    from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY

    with safetensors.torch.safe_open(model_file, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    metadata.pop("format", None)
    if metadata:
        raw = metadata.get(LORA_ADAPTER_METADATA_KEY)
        return json.loads(raw) if raw else None
    else:
        return None

def encryption_model(origin_file, encrypted_file):
    """加密函数，保存权重和metadata"""
    # 加载权重数据
    model = load_file(origin_file)
    
    # 提取metadata（不加密，直接以明文形式存储）
    metadata = _load_sft_state_dict_metadata(origin_file)
    print(metadata)
    # 将权重和metadata打包
    package = {
        'state_dict': model,
        'metadata': metadata
    }
    
    b = io.BytesIO()
    torch.save(package, b)
    b.seek(0)
    pth_bytes = b.read()
    encrypted_data = Fernet(key).encrypt(pth_bytes)
    
    with open(encrypted_file, 'wb') as fw:
        fw.write(encrypted_data)

def decryption_model(encrypt_weight_path, key=b'LxfQgPiUiXy_fxVUlLlpusLdxdEMQM_ltzY_28kXerM='):
    """解密函数，返回权重和metadata两个东西"""
    with open(encrypt_weight_path, 'rb') as fr:
        encrypted_data = fr.read()
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    package = torch.load(b, map_location='cpu', weights_only=False)
    
    # 返回两个东西：原来的checkpoint和metadata
    checkpoint = package.get('state_dict', package)  # 兼容性处理
    metadata = package.get('metadata', {})
    
    return checkpoint, metadata 

if __name__ == '__main__':
    # 需加密模型列表
    model_list = [
        '/gemini/platform/public/aigc/jiangzj/jz/pipeline/t2i/models/lightning.safetensors',
        '/gemini/platform/public/aigc/jiangzj/jz/pipeline/t2i/models/pytorch_lora_weights.safetensors'

    ]
    
    for model_name in tqdm(model_list):
        encryption_model(model_name, model_name.replace('.safetensors', '.dll.safetensors'))
