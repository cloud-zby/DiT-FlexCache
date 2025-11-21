### 创建环境
```python
$ sudo docker login harbor.telecom-ai.com.cn -u teleai -p Teleai123
$ sudo docker pull harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12:v1.1
$ sudo docker images
# REPOSITORY                                                                           TAG             IMAGE ID       CREATED        SIZE
# harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12                          v1.1            1ec25ac4820d   4 months ago   14.6GB
$ sudo docker run -it --name teleimage_v7 \
    -v /nvfile-heatstorage:/nvfile-heatstorage \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ipc=host --net host --gpus all --privileged=true \
    1ec25ac4820d bash
(base) $ cd teleai_pipe/pipelines/teleimage_v7/codes
(base) $ pip install -r requirements.txt
```

### 相关模型权重
```python
# 1. base_model                                 # 存放文生图模型
# 2. lora                                       # 存放lora模型
# 3. Qwen3-30B-A3B-Instruct-2507                # 存放大语言模型
```

### 运行相关文件
```python
(base) $ cd teleai_pipe/pipelines/teleimage_v7/codes
(base) $ python main.py        # 进行测试
```

### 相关文件说明
```python
|-- benchmarks      # 测试文件
|   `-- prompts.txt
|-- codes           # 代码文件
|   |-- configs             # 存放配置文件
|   |-- main.py             # 主文件
|   |-- pipeline            # 文生图核心模型
|   |-- preprocess          # 前处理，主要进行提示词的相关操作
|   |-- requirements.txt    # 环境配置文件
|   `-- utils               # 存放一些工具文件
|-- models          # 模型文件
`-- results         # 测试结果文件
```

### 函数调用说明
在main.py中包含了类`TeleImage`，该类核心函数为run()，主要参数为
* prompt -> 用于生成图像的提示词
* aspect_ratios -> 支持的分辨率： "1:1",  "9:16", "16:9", "4:3", "3:4", "3:2", "2:3"
* seed -> 为int类型，用于生成图像的随机种子，若为-1则随机生成
* batch_size -> 每次生成的图像数量
* num_inference_steps -> 单次推理的步数
* true_cfg_scale -> cfg


### TeleImage v7.0更新说明
本次升级显著提升了模型的画面真实感、图文一致性和文字生成能力。其主要创新包括：
1. 提升画面真实感，通过联合真实图像与AIGC图像，并加入判别模型作为奖励函数进行训练，引导模型区分真实与生成图像，从而使模型生成图像更真实、自然。通过大规模人工评估，模型胜率比达到167 : 33（V7.0 vs V6.0）；
2. 进一步提升语义理解能力，将6.0版本其中的单流DiT扩展到双流，总参数量由120亿上升到200亿参数，采用公开评测集DPG-Bench进行语义评估，总分提升4.33%（451.23 vs 432.49），性能处于业内多模态生成大模型前列；
3. 提升文字书写能力，利用文字叠加自然背景的方式到简单海报，渐进式训练模型的文字书写能力，当前文字成图率达到70.83%。