import os
import json
import uuid
import shutil
import re
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
import torchaudio
import numpy as np
from typing import List, Dict
import random
from pathlib import Path
import time
import traceback
import argparse

# TTS 相关导入
import sys
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 评估模型导入
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import wespeaker
# from Eval_Models.scripts.f0_extractor import VADTrimmer
# from Eval_Models.scripts.beats_extractor import BeatExtractor
import argparse
from tqdm import tqdm

# vc 模型导入
sys.path.append("seed-vc")
from inference import load_modelsv1,voice_conversion_inference


# ====================
# 工具类
# ====================
instruct_text_dict = {
    "HAPPY": "用开心的语气说这句话",
    "SAD": "用悲伤的语气说这句话",
    "ANGRY": "用愤怒的语气说这句话",
    "SURPRISED": "用惊讶的语气说这句话",
}
class TextUtils:
    @staticmethod
    def sanitize_text(text: str) -> str:
        """清理文本，生成安全的文件名（支持中文）"""
        return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", "_", text).strip("_")
    
    @staticmethod
    def get_text_directory(text: str) -> str:
        """获取文本对应的存储目录"""
        sanitized = TextUtils.sanitize_text(text)
        return os.path.join(Config.OUTPUT_ROOT, sanitized)

class FileManager:
    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def safe_save(path: str, data):
        """原子化保存"""
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

def calculate_score(item: Dict) -> float:
    """计算加权得分"""
    return item['emotion_score'] + item['spk_sim_score'] 

def get_emo_from_text(text):
    matches = re.findall(r"<\|([^|]+)\|>", text)
    emo = matches[1] if len(matches) > 1 else None
    return emo

def keep_only_chinese(text):
    # 保留所有汉字，去除标点、空格、字母、数字等
    return ''.join(re.findall(r'[\u4e00-\u9fff]', text))

# ====================
# 模型加载器（进程级单例）
# ====================
class TTSModelLoader:

    
    @classmethod
    def load_tts_model(cls,tts_model_path):


        tts_model = CosyVoice2(
                    tts_model_path, 
                    load_jit=False, 
                    load_trt=False, 
                    load_vllm=False, 
                    fp16=False
                )
        return tts_model

# ====================
# 评估模型（主进程单例）
# ====================
class Evaluator:
    _instance = None
    _models_loaded = False
    
    def __new__(cls,config_v1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models(config_v1)
        return cls._instance
    
    def _load_models(self,config_v1):
        if not self._models_loaded:
            print("Loading evaluation models in main process...")
            # ASR and EMO_pred模型
            self.asr_model = AutoModel(
                disable_update=True,
                model=config_v1.emo_model_path,
                trust_remote_code=True,
                remote_code=config_v1.emo_model_python_path,    
                device="cuda:0",
                disable_pbar=True
            )
            
            # 说话人模型
            self.speaker_model = wespeaker.load_model_local(config_v1.spk_model_path)
            self.speaker_model.set_device("cuda:0")
            
            # # UTMOS模型
            # self.utmos_model = utmosv2.create_model(
            #     pretrained=True,
            #     checkpoint_path="/gemini/space/ScoreTTS/checkpoints/utmos_v2/fold3_s42_best_model.pth",
            #     device="cuda:1"
            # )
            
            # # F0提取器
            # self.f0_extractor = VADTrimmer(aggressiveness=3)
            
            # # 节拍提取器
            # self.beats_extractor = BeatExtractor()
            
            # # WER评估器
            # self.wer_evaluator = evaluate.load("/gemini/space/ScoreTTS/metrics/wer.py")
            
            # # 计算参考音频属性
            # self._calculate_ref_attributes()
            
            self._models_loaded = True
    
    # def _calculate_ref_attributes(self):
    #     """计算参考音频的属性"""
    #     # 情感属性
    #     asr_result = self.asr_model.generate(
    #         input=Config.REF_SPEECH_PATH,
    #         language="auto",
    #         use_itn=False,
    #         batch_size_s=60,
    #         merge_vad=True,
    #         merge_length_s=15,
    #         ban_emo_unk=True,
    #     )
    #     self.ref_emo_possibility = asr_result[0]["pbty"].to("cpu")
        
        # F0属性
        # self.ref_f0_mean_std = self.f0_extractor.trim_and_f0_extract(
            # file_path=Config.REF_SPEECH_PATH
        # )
        
        # 节拍属性
        # self.ref_beats = self.beats_extractor.extract(
            # audio_path=Config.REF_SPEECH_PATH
        # )
    
    def evaluate(self, synthesis_items: List[Dict],config_v1) -> List[Dict]:
        """批量评估所有合成结果"""
        results = []
        
        # emotion 评估
        for i in tqdm(range(0, len(synthesis_items), config_v1.EVAL_BATCH_SIZE),desc="emotion eval..."):
            batch = synthesis_items[i:i+config_v1.EVAL_BATCH_SIZE]
            if (config_v1.TASK == "zero_shot"):
                wav_inputs = [item["wav_path"] for item in batch]
            elif (config_v1.TASK == "emotion"):
                wav_inputs = [item["vc_wav_path"] for item in batch]

            if (config_v1.TASK == "zero_shot"):
                gt_emos = [item["ref_emotion"] for item in batch]
            elif (config_v1.TASK == "emotion"):
                gt_emos = [item["need_emotion"] for item in batch]

            try:
                # ASR and emotion_pred 评估
                res_results = self.asr_model.generate(
                    input=wav_inputs,
                    language="auto",
                    use_itn=False,
                    ban_emo_unk=True,
                    output_timestamp=False,
                )

                texts_raw = [x["text"] for x in res_results]
                pred_emos = [get_emo_from_text(text) for text in texts_raw]
                emo_prob_dicts = [x['emo_prob_dict'] for x in res_results]
                # 取出每个样本的目标情感得分
                gt_emo_scores = [
                    emo_prob_dict.get(gt_emo, 0.0)  # 如果没有该情感，返回0.0
                    for gt_emo, emo_prob_dict in zip(gt_emos, emo_prob_dicts)
                ]
                # pred_texts = [rich_transcription_postprocess(text) for text in texts_raw]
                # pred_texts = [keep_only_chinese(text) for text in pred_texts]
                # refs = [keep_only_chinese(text) for text in Config.REF_TEXT]
                
                # 计算各项指标
                j = 0
                for item in batch:
                    item["emotion_score"] = gt_emo_scores[j]
                    j += 1

            except Exception as e:
                print(f"Evaluation failed for {item['wav_path']}: {str(e)}")

        # spk_sim 评估
        for i in tqdm(range(0, len(synthesis_items)),desc="spk_sim eval..."):
            item = synthesis_items[i]
            if (item["task"] == "zero_shot"):
                item["spk_sim_score"] = self.speaker_model.compute_similarity(item["wav_path"], item["ref_wav_path"])
            elif (synthesis_items[0]["task"] == "emotion"):
                item["spk_sim_score"] = self.speaker_model.compute_similarity(item["vc_wav_path"], item["ref_wav_path"])

            item["score"] = calculate_score(item)

        return synthesis_items

# ====================
# 合成函数
# ====================
def synthesize_worker(args,tts_model):
    try:
        """TTS合成工作函数"""
        text = args["text"]
        attempt = args["attempt"]
        task = args["task"]
        ref_wav_path = args["ref_wav_path"]
        ref_text = args["ref_text"]
        need_emotion = args["need_emotion"]
        ref_emotion = args["ref_emotion"]
        max_emo_item = args["max_emo_item"]
        sanitized = TextUtils.sanitize_text(text)
        text_dir = TextUtils.get_text_directory(text)
        FileManager.ensure_dir(text_dir)
        
        # 生成唯一文件名
        audio_filename = f"{sanitized}_v{attempt+1}_.wav"
        audio_path = os.path.join(text_dir, audio_filename)
        # print(f"audio_path: {audio_path}")
        
        # 加载模型并合成
        tts_model = tts_model
        
        if task == "zero_shot":
            prompt_speech_16k = load_wav(ref_wav_path, 16000)
            # 合成语音
            for i, j in enumerate(tts_model.inference_zero_shot(text, ref_text, prompt_speech_16k, stream=False)):
                torchaudio.save(audio_path, j['tts_speech'], tts_model.sample_rate)
                    # 返回合成信息
            return {
                "wav_path": audio_path,
                "text": text,
                "ref_wav_path": ref_wav_path,
                "ref_text": ref_text,
                "ref_emotion": ref_emotion,
                "task": task,
                "attempt": attempt,
                "need_emotion": None,
                "success": True,
            }
        elif task == "emotion":
            # 合成语音
            prompt_text = instruct_text_dict[need_emotion]
            # load_emotion_json = json.load(open(f'/gemini/space/jrh/programme/ScoreTTS/Emotion_data/{need_emotion}.json', "r"))
            # select_emotion_item = random.choice(load_emotion_json)
            select_emotion_item = max_emo_item
            prompt_speech_16k = load_wav(select_emotion_item["path"], 16000)

            for i, j in enumerate(tts_model.inference_instruct2(text, prompt_text, prompt_speech_16k, stream=False)):
                torchaudio.save(audio_path, j['tts_speech'], tts_model.sample_rate)
            return {
                "wav_path": audio_path,
                "text": text,
                "ref_wav_path": ref_wav_path,
                "ref_text": ref_text,
                "ref_emotion": ref_emotion,
                "task": task,
                "attempt": attempt,
                "need_emotion": need_emotion,
                "need_emotion_ref_path": select_emotion_item["path"],
                "need_emotion_ref_text": select_emotion_item["text"],
                "success": True,
            }    
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error in synthesize_worker: {tb_str}")
        # 失败返回
        return {
            "wav_path": audio_path,
            "text": text,
            "ref_wav_path": ref_wav_path,
            "ref_text": ref_text,
            "ref_emotion": ref_emotion,
            "task": task,
            "attempt": attempt,
            "success": False,
            "error": tb_str,
        }

# ====================
# 结果处理
# ====================
def process_results(results: List[Dict],config_v1):
    """处理评估结果并保存"""
    # 按文本分组
    text_groups = {}
    for item in results:
        if item["text"] not in text_groups:
            text_groups[item["text"]] = []
        text_groups[item["text"]].append(item)
    
    # 处理每个文本组
    for text, items in text_groups.items():
        text_dir = TextUtils.get_text_directory(text)
        topn_dir = os.path.join(text_dir, "topn")
        FileManager.ensure_dir(topn_dir)
        
        # 按得分排序
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        
        # 保存历史记录
        history_file = os.path.join(text_dir, "history.json")
        FileManager.safe_save(history_file, sorted_items)
        
        # 保存TopN
        topn_items = sorted_items[:config_v1.TOP_N]
        topn_data = []
        for idx, item in enumerate(topn_items, 1):
            # 复制音频文件
            dest_path = os.path.join(topn_dir, f"{TextUtils.sanitize_text(text)}_top{idx}.wav")
            # if not os.path.exists(dest_path):
            if (item["task"] == "zero_shot"):
                shutil.copy(item["wav_path"], dest_path)
            elif (item["task"] == "emotion"):
                shutil.copy(item["vc_wav_path"], dest_path)
            
            # 更新项目
            new_item = item.copy()
            new_item["wav_path"] = dest_path
            topn_data.append(new_item)
        
        # 保存TopN数据
        topn_file = os.path.join(topn_dir, "topn.json")
        FileManager.safe_save(topn_file, topn_data)
        print(f"text: {text} \noutput_wav_path: {topn_data[0]['wav_path']}")
    
    # 保存全局排名
    # global_ranking = sorted(results, key=lambda x: x["score"], reverse=True)
    # FileManager.safe_save(
    #     os.path.join(config_v1.OUTPUT_ROOT, config_v1.GLOBAL_RANK_FILE),
    #     global_ranking[:config_v1.TOP_N * len(config_v1.TEXTS_TO_SYNTHESIZE)]
    # )

# ====================
# 主流程
# ====================
def main(config_v1):
    # 初始化文件夹结构
    FileManager.ensure_dir(config_v1.OUTPUT_ROOT)
    for text in config_v1.TEXTS_TO_SYNTHESIZE:
        FileManager.ensure_dir(TextUtils.get_text_directory(text))
    
    # 步骤1: 合成
    max_emo_item = None
    if config_v1.TASK == "emotion":
        spk_model = wespeaker.load_model_local(config_v1.spk_model_path)
        spk_model.set_device("cuda:0")
        load_emotion_json = json.load(open(config_v1.emo_json_path, "r"))
        load_emotion_json = [item for item in load_emotion_json if item["emo"] == config_v1.NEED_EMOTION]
        for item in tqdm(load_emotion_json,desc="spk_sim eval..."):
            item["spk_sim_score"] = spk_model.compute_similarity(item["path"], config_v1.REF_SPEECH_PATH)
            print(f"spk_sim_score: {item['spk_sim_score']}")
        # 根据这个score 排序
        sorted_emo_spk_sim_scores = sorted(load_emotion_json, key=lambda x: x["spk_sim_score"], reverse=True)
        max_emo_item = sorted_emo_spk_sim_scores[0]
        # 删除模型
        del spk_model
        torch.cuda.empty_cache()# 清显存
        gc.collect() 

    print(f"Starting synthesis with {config_v1.MAX_PROCESSES} processes...")
    tts_model = TTSModelLoader.load_tts_model(config_v1.tts_model_path)
    synthesis_tasks = []
    for text in config_v1.TEXTS_TO_SYNTHESIZE:
        for attempt in range(config_v1.SYNTHESIS_REPEATS):
            synthesis_tasks.append({"text": text, 
                                    "attempt": attempt,
                                    "task": config_v1.TASK,
                                    "ref_wav_path": config_v1.REF_SPEECH_PATH,
                                    "ref_text": config_v1.REF_TEXT,
                                    "ref_emotion": config_v1.REF_EMOTION,
                                    "need_emotion": config_v1.NEED_EMOTION,
                                    "max_emo_item": max_emo_item,
                                    })
    # with Pool(processes=config_v1.MAX_PROCESSES) as pool:
    #     synthesis_results = pool.map(synthesize_worker, synthesis_tasks)
    synthesis_results_list = []
    for task in synthesis_tasks:
        synthesis_results = synthesize_worker(task,tts_model)
        synthesis_results_list.append(synthesis_results)
    # 过滤掉失败的
    success_results = [r for r in synthesis_results_list if r.get("success")]
    print(f"Completed {len(success_results)} synthesis tasks")
    # 删除模型
    del tts_model
    torch.cuda.empty_cache()# 清显存
    gc.collect() 
    # 如果任务是emotion，则需要进行vc
    if config_v1.TASK == "emotion":
        # 加载vc模型
        print(config_v1.rmvpe_path)
        print(type(config_v1.rmvpe_path))

        vc_models = load_modelsv1(dit_checkpoint_path=config_v1.dit_checkpoint_path,
                        dit_config_path=config_v1.dit_config_path, 
                        rmvpe_path=config_v1.rmvpe_path, 
                        campplus_path=config_v1.campplus_path,
                        bigvgan_name=config_v1.bigvgan_name,
                        whisper_name=config_v1.whisper_name)
        for item in success_results:
            source_wav_path = item["wav_path"]
            target_wav_path = item["ref_wav_path"]
            old_path = Path(item["wav_path"])
            output_wav_path = str(old_path.with_name(f"vc_{old_path.name}"))
            voice_conversion_inference(vc_models,source_wav_path,target_wav_path,output_wav_path,auto_f0_adjust=True)
            item["vc_wav_path"] = output_wav_path
        print(f"Completed {len(success_results)} voice conversion tasks")


    # 步骤2: 主进程评估
    print("Starting evaluation in main process...")
    evaluator = Evaluator(config_v1)
    evaluated_results = evaluator.evaluate(success_results,config_v1)
    
    # 步骤3: 处理结果
    print("Processing results...")
    process_results(evaluated_results,config_v1)
    print("Processing completed!")

def parse_args():
    parser = argparse.ArgumentParser(description="Text to Speech Synthesis")
    
    # 任务类型
    parser.add_argument(
        "--task",
        type=str,
        choices=["zero_shot", "emotion"],
        required=True,
        help="任务类型"
    )
    
    # 情感类型
    parser.add_argument(
        "--need-emotion",
        type=str,
        choices=["HAPPY", "SAD", "ANGRY", "SURPRISED"],
        default=None,
        help="情感类型 (仅在 task=emotion 时需要)"
    )
    
    # 参考音频路径
    parser.add_argument(
        "--ref-speech-path",
        type=str,
        required=True,
        help="参考音频路径 (.wav 或 .mp3 文件)"
    )
    
    # 输出目录
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tts_results",
        help="输出目录路径"
    )
    
    # 要合成的文本
    parser.add_argument(
        "--texts-to-synthesize",
        type=str,
        required=True,
        help="需要合成的文本"
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if args.task == "emotion" and args.need_emotion is None:
        parser.error("当 task=emotion 时，必须指定 --need-emotion")
    
    # 验证参考音频文件是否存在且格式正确
    if not os.path.exists(args.ref_speech_path):
        parser.error(f"参考音频文件不存在: {args.ref_speech_path}")
    
    return args
if __name__ == "__main__":
    args = parse_args()

    
    # ====================
    # 配置中心
    # ====================
    class Config:
        TASK = args.task # 任务类型 "zero_shot" or "emotion"
        NEED_EMOTION = args.need_emotion # 情感类型 "HAPPY" "SAD" "ANGRY" "SURPRISED"
        TOP_N = 2        # 保留的TopN数量
        MAX_PROCESSES = 1  # TTS进程数
        REF_SPEECH_PATH = args.ref_speech_path
        REF_TEXT = None
        OUTPUT_ROOT = args.output_dir
        SYNTHESIS_REPEATS = 5  # 每个文本合成次数
        EVAL_BATCH_SIZE = 2# 评估批量大小
        TEXTS_TO_SYNTHESIZE = [args.texts_to_synthesize]
        #vc model path
        dit_checkpoint_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/Plachta-Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
        dit_config_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/Plachta-Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
        rmvpe_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/lj1995-VoiceConversionWebUI/rmvpe.pt"
        campplus_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/funasr-campplus/campplus_cn_common.bin"
        bigvgan_name="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/nvidia-bigvgan_v2_44khz_128band_512x"
        whisper_name="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/whisper-small"
        #spk model path
        spk_model_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/voxblink2_samresnet34"
        #tts model path
        tts_model_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/CosyVoice2-0.5B"
        #emo model path
        emo_model_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
        #emo_model_py_path
        emo_model_python_path="Eval_Models/SenseVoice-main/model.py"
        #emotion json path
        emo_json_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/Emotion_data/merge_merged.json"

    # ASR and EMO_pred模型
    emo_model = AutoModel(
        disable_update=True,
        model=Config.emo_model_path,
        trust_remote_code=True,
        remote_code=Config.emo_model_python_path,    
        device="cuda:0",
        disable_pbar=True
    )
    # 模型生成推理结果
    res = emo_model.generate(
            input=Config.REF_SPEECH_PATH,
            language="zh",
            use_int=False,
            ban_emo_unk=True,
            output_timestamp=False,
        )
    text = rich_transcription_postprocess(res[0]["text"])
    text = keep_only_chinese(text)
    matches = re.findall(r"<\|([^|]+)\|>", res[0]["text"])
    ref_emo = matches[1] if len(matches) > 1 else None

    print(f"ref_emo: {ref_emo}")
    print(f"text: {text}")
    
    config_v1 = Config()
    config_v1.REF_EMOTION = ref_emo
    config_v1.REF_TEXT = text
    # 删除模型
    del emo_model, res
    torch.cuda.empty_cache()# 清显存
    gc.collect() 

    main(config_v1)
