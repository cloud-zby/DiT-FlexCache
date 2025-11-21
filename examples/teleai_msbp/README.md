### 创建环境
```python
$ sudo docker login harbor.telecom-ai.com.cn -u teleai -p Teleai123
$ sudo docker pull harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12:v1.1
$ sudo docker images
# REPOSITORY                                                                           TAG             IMAGE ID       CREATED        SIZE
# harbor.telecom-ai.com.cn/teleai-t2v/ubuntu22_torch25_cuda12                          v1.1            1ec25ac4820d   4 months ago   14.6GB
$ sudo docker run -it --name teleai_msbp \
    -v /nvfile-heatstorage:/nvfile-heatstorage \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ipc=host --net host --gpus all --privileged=true \
    1ec25ac4820d bash

$ pip install --force-reinstall --no-deps /nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/packages/accelerate-1.7.0.dev0/dist/accelerate-1.7.0.dev0-py3-none-any.whl
$ pip install --force-reinstall --no-deps /nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/packages/diffusers-0.31.0/dist/diffusers-0.31.0-py3-none-any.whl
$ pip install --force-reinstall --no-deps /nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/packages/transformers-4.51.0/dist/transformers-4.51.0-py3-none-any.whl
$ pip install --force-reinstall --no-deps /nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/packages/sbp/dist/sbp-0.1-py3-none-any.whl
$ cd teleai_pipe/pipelines/teleai_msbp
$ pip install -r requirements.txt
```

### 算法流程
- **用户输入：**  
  a. 图片描述 -> 对应参数 `prompt`  
  
  b. 需要控制的人脸图片列表 -> 对应参数 `ref_faces`  
  
  c. 需要控制的人体(服装、整体造型)图片列表 -> 对应参数 `ref_dressing`  
  
  d. 需要编辑的图片 -> 对应参数 `ref_edit`  
  
  e. 需要修改的区域 -> 对应参数 `bbox`  
  
  f. 需要控制的人体姿态 -> 对应参数 `pose_keypoints`  
  若干数量的 24 或 134 关键点列表，如果输入 134 关键点则使用 dwpose  
  
- **模型输出：**  
  
  ◦ 生成图片/编辑图片 
  
### 函数
#### 调用实例
```
cd /nvfile-heatstorage/fengzr/projects/teleai_pipe/teleai_pipe/pipelines/teleai_msbp 

bash msbp/run_flux_infer7_with_pose.sh
# 需要两张H100, 结果在output
```

### 类与方法

**文件名：infer7_with_pose.py**

**函数 gen_single**

输入参数

| 参数名           | 类型                            | 可选值及含义                     |  
|------------------|--------------------------------|---------------------------------|  
| `prompt`         | `str`                          | 图片描述（英文）                 |  
| `ref_faces`      | `list` of `str`                | 人脸图片路径列表                 |  
| `ref_dressing`   | `list` of `str`                | 服装图片路径列表                 |  
| `ref_edit`       | `list` of `str`                | 待编辑图片路径列表               |  
| `bboxes`         | `T.List[T.List[bbox]]`         | 对应每张待编辑图片有若干 `bbox` |  
| `pose_keypoints` | `T.List[T.List[kps]]`           | 对应每张待编辑图片有若干关键点列表，与 `bbox` 一致 |  
| `prompt_for_bboxes` | `T.List[T.List[str]]`         | 对应每个 `bbox` 的描述字符串    |  

注释：

类型定义：
bbox = List[int] （格式为 [x1, y1, x2, y2]）
kps = List[point]
point = List[int] （格式为 [x, y]）

输出参数

| 参数名          | 类型                      | 可选值及含义                                         |  
|-----------------|---------------------------|------------------------------------------------------|  
| `generated_images` | `list[PIL.Image]`          | 生成的图片列表                                       |  
| `ref_bboxes`      | `T.List[T.List[bbox]]` \| `None` | 人脸`bbox`，如果生成图片没有检测到人脸，则为 `None` |  





