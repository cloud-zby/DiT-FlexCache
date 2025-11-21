import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "teleai_pipe/pipelines/teleai_ai2v/clearvoice"))
from clearvoice import ClearVoice
import numpy as np
import soundfile as sf
import json
import subprocess

def audio_subtraction(file1, file2, output_file):
    data1, samplerate1 = sf.read(file1)
    data2, samplerate2 = sf.read(file2)

    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    result_data = data1 - data2

    sf.write(output_file, result_data, samplerate1)

def main_clearvoice(config, video_index, clearvoice_model, data=None):
    output_audio_dir = os.path.join(config['target_path'], config['version'], config['audio_result_path'], "raw")
    os.makedirs(output_audio_dir, exist_ok=True)
    output_audio_origin = os.path.join(output_audio_dir, f"{video_index}_origin.wav")
    output_audio_human = os.path.join(output_audio_dir, f"{video_index}_human.wav")
    output_audio_bg = os.path.join(output_audio_dir, f"{video_index}_bg.wav")

    if os.path.isfile(output_audio_origin) and os.path.isfile(output_audio_human) and os.path.isfile(output_audio_bg):
        print(f"skip video >>> {video_index}")
        return

    video_path = os.path.join(config['target_path'], config['version'], "video", "data", f"{video_index}.mp4")
    try:
        # extract raw audio using ffmpeg
        ffmpeg_bin_str = config["ffmpeg"]
        # command_extract_audio = \
        #         # f"{ffmpeg_bin_str} -i {video_path} -y -q:a 0 -map a -ar 44100 {output_audio_origin}"
        # subprocess.call(command_extract_audio, shell=True)
        subprocess.run(
            [f'{ffmpeg_bin_str}', '-i', video_path, '-loglevel', 'quiet', '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', output_audio_origin], \
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # human voice enhancement
        #  Process an input waveform and return the enhanced output waveform
        output_wav = clearvoice_model(input_path=output_audio_origin)
        clearvoice_model.write(output_wav, output_path=output_audio_human)

        # raw_audio - human_audio = bg_audio
        audio_subtraction(output_audio_origin, output_audio_human, output_audio_bg)
    except Exception as e:
        # TODO: dump error message
        print(e)
        pass

if __name__ == '__main__':
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    
    config_path = "/workspace/teleAI-T2V/new_data_ppl_v1.0/config_test.json"
    config = json.load(open(config_path))
    sub_config = config["datalist"][0]
    sub_config.update(config["modelinfo"])
    sub_config.update(config["tools"])
    main_clearvoice(sub_config, 888, myClearVoice)