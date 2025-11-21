# video to sound

Give videos, the model will generate the corresponding sound(no speech and bgm).

## Inference

``` bash
python demo.py
```

The model checkpoints path is define in demo.py: 
``` python
@dataclass
class Config:
    model_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/sound_effect_with_blank_voice_vgg_90000.pth"
    audio_vae_path: str = "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/v1-44.pth"
    vocoder_ckpt_path: str = (
        "/nvfile-heatstorage/sgh/model/nvidia-bigvgan_v2_44khz_128band_512x"
    )
    video_encoder_path: str = (
        "/nvfile-heatstorage/sgh/model/MMaudio/ext_weights/synchformer_state_dict.pth"
    )
    text_encoder_path: str = "/nvfile-heatstorage/sgh/model/t5-base"
    text_max_length: int = 64
    imagebind_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/imagebind_huge.pth"
    bpe_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/bpe_simple_vocab_16e6.txt.gz"
    vidmuse_ckpt_path: str = "/nvfile-heatstorage/sgh/share/tele_sound/music"
```

The input video, prompt and output dir is defined by:
```
prompt = "piano"
input_video_file = Path("/nvfile-heatstorage/sgh/project/teleai_pipe/eggs/input/1.mp4")
output_dir = "/nvfile-heatstorage/sgh/project/teleai_pipe/examples/video2sound/output/"
```

