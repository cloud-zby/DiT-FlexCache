
## 性能测试

|task| example file | device |attn| resolution | frame num| fps | video length | cfg | teacache | infer cost | gpu mem|
|---|---|---|---|---|---|---|---|---|---|---|---|
|H100文生视频|t2v_80G.py| h100 *1|sage88| 720p|81| 15| 5s|False|0.15|2min34s|50G
|H100文生视频|t2v_80G.py| h100 *1|sage88| 720p|81| 15| 5s|True|0.15|5min20s|50G
|4090文生视频|t2v_parallel.py|4090*4|sage88|720p|81|15|5s|False|0.15|5min20s|20G
|H100图生视频|i2v_80G.py| h100 *1|sage88| 720p|81| 15| 5s|False|0.15|2min34s|50G
|H100图生视频|i2v_80G.py| h100 *1|sage88| 720p|81| 15| 5s|True|0.15|5min20s|50G
|4090图生视频|i2v_parallel.py|4090*4|sage88|720p|81|15|5s|False|0.15|5min20s|20G
|H100首尾帧生视频|FL_i2v_80G.py| h100 *1|sage88| 720p|49| 8| 5s|True|0.15|3min|45G
|4090首尾帧生视频|FL_i2v_parallel.py| 4090 *4|sage88| 720p|49| 8| 5s|True|0.15|7min|20G
|H100动作生视频|s2v_diffusers_80G_nobox_continue_demo|h100*1|sage88|720p|45*2|15|6s|True|0.15|5min|45G
|4090动作生视频|s2v_diffusers_80G_nobox_continue_demo|4090*1|sage88|720p|45*2|15|6s|True|0.15|20min|20G