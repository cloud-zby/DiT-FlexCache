import librosa

class BeatExtractor:
    @staticmethod
    def extract(audio_path, sr=16000, hop_length=160):
        """
        提取音频文件中的节拍时间点（单位：秒）

        参数:
            audio_path (str): 音频文件路径
            sr (int): 采样率，默认 16000
            hop_length (int): 每帧分析的样本数，默认 160

        返回:
            List[float]: 节拍速度
        """
        y, _ = librosa.load(audio_path, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        # beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        return tempo

if __name__ == "__main__":

    wav_path = "/gemini/space/ScoreTTS/fine_grained_control_0.wav"

    beats = BeatExtractor.extract(wav_path)

    # 打印节拍时间点
    print("检测到的节拍时间点（秒）：", beats)