import time
import sys
sys.path.append("/gemini/space/jrh/programme/ScoreTTS/seed-vc")
from inference import load_modelsv1,voice_conversion_inference
time_start = time.time()
models = load_modelsv1()
time_end = time.time()
print(f"load models time: {time_end - time_start} seconds")

source_wav_path = "/gemini/space/jrh/programme/ScoreTTS/tts_results_ANGRY/这些年你背着天尊做了什么_当我不知道吗/这些年你背着天尊做了什么_当我不知道吗_v4_.wav"
target_wav_path = "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3"
output_wav_path = "/gemini/space/jrh/programme/ScoreTTS/vc_test.wav"
output_wav_path1 = "/gemini/space/jrh/programme/ScoreTTS/vc_test1.wav"
time_start = time.time()
voice_conversion_inference(models,source_wav_path,target_wav_path,output_wav_path,auto_f0_adjust=False)
time_end = time.time()
print(f"voice conversion time: {time_end - time_start} seconds")
time_start = time.time()
voice_conversion_inference(models,source_wav_path,target_wav_path,output_wav_path1,auto_f0_adjust=True)
time_end = time.time()
print(f"voice conversion time: {time_end - time_start} seconds")