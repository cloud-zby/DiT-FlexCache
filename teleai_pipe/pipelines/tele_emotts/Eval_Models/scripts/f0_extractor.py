import numpy as np
import pyworld
import librosa
import matplotlib.pyplot as plt


import webrtcvad
import librosa
import numpy as np
import soundfile as sf

class VADTrimmer:
    def __init__(self, aggressiveness=2, frame_ms=10):
        self.aggressiveness = aggressiveness  # 0~3，数值越大越“激进”地裁剪静音
        self.frame_ms = frame_ms             # 通常为 10、20 或 30ms
        self.target_sr = 16000               # WebRTC VAD 只支持 16kHz

    @staticmethod
    def float2pcm(wav_float):
        return np.clip(wav_float * 32768, -32768, 32767).astype(np.int16)

    def frame_generator(self, wav_pcm, frame_size):
        for i in range(0, len(wav_pcm) - frame_size + 1, frame_size):
            yield wav_pcm[i:i + frame_size]

    def detect_voiced_region(self, y_orig, sr_orig):
        # 重采样到 16kHz
        y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=self.target_sr)
        pcm = self.float2pcm(y_16k)

        # 初始化 VAD
        vad = webrtcvad.Vad(self.aggressiveness)
        frame_size = int(self.target_sr * self.frame_ms / 1000)
        frames = list(self.frame_generator(pcm, frame_size))
        flags = [vad.is_speech(f.tobytes(), self.target_sr) for f in frames]

        # 获取有效语音区域
        voiced = [i for i, f in enumerate(flags) if f]
        if not voiced:
            return 0, len(y_orig)  # 全静音

        start_time = voiced[0] * self.frame_ms / 1000.0
        end_time = (voiced[-1] + 1) * self.frame_ms / 1000.0

        start_sample = int(start_time * sr_orig)
        end_sample = min(int(end_time * sr_orig), len(y_orig))

        return start_sample, end_sample

    def trim(self, y, sr):
        start, end = self.detect_voiced_region(y, sr)
        return y[start:end], start, end

    def trim_file(self, input_path, output_path=None):
        y, sr = librosa.load(input_path, sr=None)
        y_trimmed, start, end = self.trim(y, sr)

        if output_path:
            sf.write(output_path, y_trimmed, sr)

        return {
            "y_trimmed": y_trimmed,
            "sr": sr,
            "start_sample": start,
            "end_sample": end,
            "start_time": start / sr,
            "end_time": end / sr,
        }
    def f0_extract(self,x,fs):
        # input,x:输入单条音频 shape [n] list
        # input,fs:采样率
        # return,f0_refined, f0, shape [n1] list
        # return,time_axis, 时间轴,   shape [n1] list

        # Step 1: 使用 DIO 进行初步基频估计
        f0, time_axis = pyworld.dio(x.astype(np.float64), fs)  # DIO 基频提取（粗略）每帧5ms 

        # Step 2: 使用 StoneMask 进行基频精细化处理（推荐）
        f0_refined = pyworld.stonemask(x.astype(np.float64), f0, time_axis, fs)  # 精细化F0
        return f0_refined,time_axis
    def trim_and_f0_extract(self,file_path):
            # input,file_path:输入单条音频路径 
            # input,fs:采样率
            # return,f0_refined, f0, shape [n1] list
            # return,time_axis, 时间轴,   shape [n1] list
        info = self.trim_file(file_path)
        x,fs = info["y_trimmed"],info["sr"]
        f0_refined,time_axis = self.f0_extract(x,fs)
        mean_std = [np.mean(f0_refined),np.std(f0_refined)]
        return mean_std
    


# 读取音频文件
# wav_path = "/gemini/space/ScoreTTS/zero_shot_1.wav"
# x, fs = librosa.load(wav_path, sr=None)  # 保持原采样率
def f0_extract(x,fs):
    # input,x:输入单条音频 shape [n] list
    # input,fs:采样率
    # return,f0_refined, f0, shape [n1] list
    # return,time_axis, 时间轴,   shape [n1] list

    # Step 1: 使用 DIO 进行初步基频估计
    f0, time_axis = pyworld.dio(x.astype(np.float64), fs)  # DIO 基频提取（粗略）每帧5ms 

    # Step 2: 使用 StoneMask 进行基频精细化处理（推荐）
    f0_refined = pyworld.stonemask(x.astype(np.float64), f0, time_axis, fs)  # 精细化F0
    return f0_refined,time_axis

def calculate_euclidean_distances(A, reference_point):
    """
    计算矩阵 A 中每个点与参考点之间的欧氏距离。
    
    参数:
        A (numpy.ndarray): 形状为 [2] 的矩阵，。
        reference_point (numpy.ndarray): 参考点，形状为 [2]。
    
    返回:
        numpy.ndarray: 每个点与参考点的欧氏距离，形状为 [b]。
    """
    return np.linalg.norm(A - reference_point, axis=0)


if __name__ == "__main__":
 
    f0_refined,time_axis = f0_extract(x,fs=fs)         # 提取单个音频的f0 
    mean_std = [np.mean(f0_refined),np.std(f0_refined)] # 单条音频的mean 和 std
    mean_std_list = mean_std      
    reference_mean_std = [100,100]
    a = calculate_euclidean_distances(mean_std_list,reference_point=np.array(reference_mean_std))   # 统一计算欧式距离
    print(a)
    print(f"Mean f0: {np.mean(f0_refined):.3f} ± {np.std(f0_refined):.3f}")   

# 可视化 F0
# plt.plot(time_axis, f0_refined, label='Refined F0')
# plt.xlabel("Time [s]")
# plt.ylabel("Frequency [Hz]")
# plt.title("F0 contour extracted by WORLD (DIO + StoneMask)")
# plt.grid()
# plt.legend()
# plt.savefig("/gemini/space/ScoreTTS/f0_plot.png", dpi=300, bbox_inches='tight')  # dpi 控制清晰度
    
# trimmer = VADTrimmer(aggressiveness=3)

# # 从文件裁剪并保存
# info = trimmer.trim_file("/gemini/space/ScoreTTS/zero_shot_1.wav")        # 单条 音频进行trim
# x,fs = info["y_trimmed"],info["sr"]
# print(f"裁剪时间段: {info['start_time']:.2f}s - {info['end_time']:.2f}s")
