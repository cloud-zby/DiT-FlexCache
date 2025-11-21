# 🎙️ Emotion TTS: Reference-Based Speech Synthesis with Emotion Control

**Emotion TTS** is a reference-based speech synthesis tool that supports both zero-shot speaker cloning and controllable emotional speech generation. By providing a reference audio and a target text, this tool synthesizes speech that matches the speaker's voice and optionally conveys a desired emotion.

---

## 🚀 Features

- 🔈 **Zero-shot voice cloning**  
  Generate speech using the speaker identity from a reference audio without needing additional training data.

- 🎭 **Emotion-conditioned synthesis**  
  Generate speech with one of four target emotions:  
  `HAPPY`, `SAD`, `ANGRY`, `SURPRISED`.

- 📝 **Flexible text input**  
  Synthesize any custom sentence provided through the command-line.

- 📂 **Custom output directory**  
  Save results to your preferred directory.

---

## ⚙️ Command-line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--task` | `str` | Task type: `zero_shot` or `emotion` (required) |
| `--need-emotion` | `str` | Emotion type (required only for `emotion` task). Choices: `HAPPY`, `SAD`, `ANGRY`, `SURPRISED` |
| `--ref-speech-path` | `str` | Path to reference audio (`.wav` or `.mp3`) (required) |
| `--texts-to-synthesize` | `str` | The text to be synthesized (required) |
| `--output-dir` | `str` | Output directory (default: `tts_results/`) |

---

## 🧪 Inference Examples

### Conda Environment Configuration

conda create -n scoretts python==3.12
conda activate scoretts
cd teleai_pipe/examples/emotion_tts
pip install -r requirements_emotts.txt --no-deps
cd /nvfile-heatstorage/sgh/share/jrh/tele_emotts/Eval_Models/wespeaker-master
pip install -e .
cd teleai_pipe
bash examples/emotion_tts/run.sh

### The model checkpoints path is define in main_multiprocessing.py: 
``` python
    class Config:
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
        #emotion json path
        emo_json_path="/nvfile-heatstorage/sgh/share/jrh/emotts_checkpoints/Emotion_data/merge_merged.json"
```

### 1️⃣ Zero-shot voice cloning

Generate speech with the speaker identity from the reference audio:

```bash
python main_multiprocessing.py \
    --task zero_shot \
    --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" \
    --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" \
    --output-dir tts_results_zeroshot
```
### 2️⃣ Emotion-conditioned synthesis

```bash
python main_multiprocessing.py \
    --task emotion \
    --need-emotion HAPPY \
    --ref-speech-path "/gemini/space/jrh/programme/ScoreTTS/2月14日 (2)(5).MP3" \
    --texts-to-synthesize "这些年你背着天尊做了什么，当我不知道吗。" \
    --output-dir tts_results_HAPPY
```




