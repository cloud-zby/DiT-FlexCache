## 性能测试

|task| example file | device |attn| resolution | frame num| fps | video length | cfg | teacache | magcache| infer cost | gpu mem|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|H100图生视频|wan_14b_image_to_long_video.py| h100 *1|sage88| 720p|161| 16| 10s|False|0.3|False|11min|70G
|H100图生视频|wan_14b_image_to_video.py| h100 *1|sage88| 720*1080|81| 16| 5s|True|0|True|4min15s + 13s|50G
|H100图生视频|wan_14b_image_to_video.py| h100 *1|sage88| 720*1080|81| 16| 5s|True|0|False|10min30s + 13s|50G
|H100图生视频|wan_14b_image_to_video.py| h100 *1|sage88| 720*1080|81| 16| 5s|True|0.3|False|5min30s + 13s|50G