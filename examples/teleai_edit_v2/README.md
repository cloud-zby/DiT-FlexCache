### 创建环境
```python
$ sudo docker login harbor.telecom-ai.com.cn -u teleai -p Teleai123
$ sudo docker pull harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12:v1.1
$ sudo docker images
# REPOSITORY                                                                           TAG             IMAGE ID       CREATED        SIZE
# harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12                          v1.1            1ec25ac4820d   4 months ago   14.6GB
$ sudo docker run -it --name teleedit_v2 \
    -v /nvfile-heatstorage:/nvfile-heatstorage \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ipc=host --net host --gpus all --privileged=true \
    1ec25ac4820d bash
(base) $ cd teleai_pipe/pipelines/teleai_edit_v2
(base) $ pip install -r requirements.txt
```

### 相关模型权重
```python
# 1. FLUX.1-dev                                 # 存放底模
# 2. FLUX.1-dev-Controlnet-Inpainting-Beta      # 存放control inpainting模型
# 3. FLUX.1-Fill-dev                            # 存放inpanting模型
# 4. lora/checkpoint-35000                      # 存放用于编辑消除的lora
# ~5. lora/local_lora16.safetensors             # 存放用于编辑增删的lora~
# 6. nlp_csanmt_translation_zh2en               # 存放翻译模型
```

### 运行相关文件
```python
(base) $ cd teleai_pipe/pipelines/teleai_edit_v2/codes
(base) $ python main.py        # 进行测试
```

### 相关文件说明
```python
|-- benchmarks      # 测试文件
|   |-- add
|   |   |-- 000002
|   |   |   |-- 000002-input.png
|   |   |   |-- 000002-mask.png
|   |   |   `-- object.txt
|   |   `-- ...
|   |-- remove
|   `-- replace
|-- codes           # 代码文件
|   |-- assets
|   |   `-- fonts           # 存放字体文件
|   |-- configs             # 存放配置文件
|   |-- demo.py             # demo文件
|   |-- main.py             # 主文件
|   |-- pipeline            # 增删改操作
|   |-- postprocess         # 后处理，主要用于生成测试对比文件
|   |-- preprocess          # 前处理，主要进行图像、掩膜、文本的预处理
|   |-- requirements.txt    # 环境配置文件
|   `-- utils               # 存放一些工具文件
|-- models          # 模型文件
`-- results         # 测试结果文件
```

### 函数调用说明
在main.py中包含了类`TeleEdit`，该类核心函数为run()，主要参数为
* image_path -> 用于编辑的的原始图像
* mask_path ->  掩膜图像
* prompt -> 在增加/替换操作中主要表示增加/替换的内容，可支持中文/英文，要求为一个短语，而非一句话。
* seed -> 为int类型，表示随机种子，若为-1则随机生成


### TeleEdit v2.0更新说明
1. 新增擦除操作
2. 更新增加、替换操作底模
3. 新增图像、mask、文本预处理操作，新增演示操作

### TeleEdit v2.1更新说明
1. 新增指令解析模块，分析提示词操作类型。（注：仅对第一个新增/替换/删除进行操作）
2. 支持名词性短语，指令两种输入样式。对于名词性短语，若有则进行新增/替换操作，若无则进行删除操作。
3. 删除原有operation_id参数。