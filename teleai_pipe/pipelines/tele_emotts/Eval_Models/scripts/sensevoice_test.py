




import os
import re
import json
import torch
import argparse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from tqdm import tqdm  # 用于显示进度条
import jiwer
from jiwer import Compose
import time


model_dir="/gemini/space/jrh/programme/ScoreTTS/checkpoints/SenseVoiceSmall/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
remote_code="/gemini/space/jy/SenseVoice/model.py"


cuda_device="cuda:1"          # 记得更改

# --------------------------
# 加载模型
# --------------------------
m = AutoModel(
    disable_update=True,
    model=model_dir,
    trust_remote_code=True,
    disable_pbar=True,
    remote_code=remote_code,
    device=cuda_device if torch.cuda.is_available() else "cpu"
)

def keep_only_chinese(text):
    # 保留所有汉字，去除标点、空格、字母、数字等
    return ''.join(re.findall(r'[\u4e00-\u9fff]', text))


input1_wav = "/gemini/space/jrh/datasets/libritts/LibriTTS_zh/train-clean-100/6563/285357/6563_285357_000002_000004_zh.wav"
input2_wav = "/gemini/space/jrh/datasets/libritts/LibriTTS_zh/train-other-500/1414/130538/1414_130538_000123_000000_zh.wav"
input_list = [input1_wav, input2_wav]
ref_text = "这艘飞船长四百十四英尺，直径三十八英尺，配备了十七个氢气球，总容量为三十六万七千立方英尺，此外还装有两个八十五马力的发动机驱动四个螺旋桨，总重量为九吨。"

emo_synthesis = "SURPRISED"
# 模型生成推理结果
start_time = time.time()
    # 模型生成推理结果
res = m.generate(
        input=input_list,
        language="zh",
        use_int=False,
        ban_emo_unk=True,
        output_timestamp=False,
    )

    # 逐个音频文件处理生成的结果
print(res)
end_time = time.time()
print(f"推理耗时: {end_time - start_time:.3f} 秒")
   

# 提取文本内容并后处理
pred_text = rich_transcription_postprocess(res[0]["text"])

# 使用正则匹配提取情感标签，例如 <|angry|> <|happy|>
matches = re.findall(r"<\|([^|]+)\|>", res[0]["text"])
emo = matches[1] if len(matches) > 1 else None

# 获取情感可能性值（浮点数）
possibility = res[0]["pbty"].item()

print(f"pred_text: {pred_text}")
print(f"pred_emo: {emo}")
print(f"pred_possibility: {possibility}")
print(f"res[0]: {res[0]}")
print(f"emo_prob_dict: {res[0]['emo_prob_dict']}")
print(f"{emo_synthesis}_prob: {res[0]['emo_prob_dict'][emo_synthesis]}")




# 清洗
clean_pred_text = keep_only_chinese(pred_text)
clean_ref_text = keep_only_chinese(ref_text)

cer = jiwer.cer(clean_ref_text, clean_pred_text)


print(f"clean_pred_text: {clean_pred_text}")
print(f"clean_ref_text: {clean_ref_text}")
print(f"cer: {cer}")
        

