# TeleAiEfficient 视频生成缓存复用推理框架

本项目基于TeleAIPipe实现视频生成推理场景的缓存复用，通过多机分布式存储，保存并反复用请求间的缓存，在不对视频生成质量造成明显影响前提下提升推理性能，降低端到端时延。

## 环境搭建
### 1. 克隆项目

```bash
git clone ${USER}@code.srdcloud.cn:29418/P24HQASYF0004/AI-Infra/teleai_efficient
cd teleai_efficient && git checkout feat/cache
```
### 2. 创建虚拟环境
```bash
apt install python3.10-venv
python -m venv dit-cache
source dit-cache/bin/activate
```
### 3. 安装项目

```bash
# 安装TeleAIPipe
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install -v -e .
```

### 4. 安装对应依赖

```bash
# 安装TeleFuser依赖
cd ../TeleFuser && pip install -e .
```

## 快速推理

在运行前，请设置以下环境变量：

```bash
export FLEXCACHE_KV_CONFIG=examples/flexcache/KVstore.yaml
export QDRANT_URL=http://127.0.0.1:6333
export TELEAIPIPE_CACHE_DIR=./flexcache
```

- `FLEXCACHE_KV_CONFIG`：UnifiedKV 的配置文件路径
- `QDRANT_URL`：Qdrant 服务地址（向量检索可选）
- `TELEAIPIPE_CACHE_DIR`：本地缓存目录（索引/元数据）

### 启动 KV 组件（etcd + master）

```bash
sudo apt-get install etcd

cd ../teleai_efficient/
bash examples/flexcache/start_etcd.sh
python examples/flexcache/start_master.py
```

- 如遇 “member already exists in cluster” 冲突，请停止此前同一 `instance_key` 的 master，或修改 `examples/flexcache/KVstore.yaml` 中的 `instance_key` 为新值。

### 启动 Qdrant（向量检索）

```bash
# 若有本地二进制
cd examples/flexcache && ./qdrant
```
启动后可通过 `curl http://127.0.0.1:6333/` 快速探测。

###  启动服务端（API Server）

```bash
python teleai_pipe/api_server.py \
  -pp examples/flexcache/wan22_14b_text_to_video_prompt_enhance_h100.py \
  --task t2v \
  --host 0.0.0.0 --port 8000 -g 2
```

- `-pp` 指向 T2V pipeline 的示例脚本；`--task` 使用 `t2v`。
- 并行卡数（`-g`）按需调整。

### 发送请求（示例）

```bash
curl -X POST "http://localhost:8000/v1/tasks/" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A stylish little girl gently caressing her dog in a sunny backyard",
       "negative_prompt": "",
       "target_video_length": 81,
       "aspect_ratio": "16:9",
       "seed": 42
     }'
```

返回体包含 `task_id` 与 `save_video_path`。后续可查询状态与下载：

```bash
curl "http://localhost:8000/v1/tasks/{task_id}/status"
curl "http://localhost:8000/v1/tasks/{task_id}/result" -o output.mp4
```

### 文件说明

- 本地索引：`{TELEAIPIPE_CACHE_DIR}/index.pkl`（prompt → cache_id/saved_steps）
- 元数据：`{TELEAIPIPE_CACHE_DIR}/cache_meta.pkl`（大小、访问次数、LRU 信息）
- KV 键：`{cache_id}_step{step}`（latent），以及可选的 `{cache_id}_object_mask`、`{cache_id}_background_mask`、`{cache_id}_video_tensor`

### 配置说明

- 关键步骤（key steps）：可用环境变量 `TELEAI_KEY_STEPS="0,1,5,10,15,20,25"` 覆盖，或在构造 `FlexCache` 时传入。
- 缺省行为：当 Qdrant 或 UnifiedKV 不可用时，引擎会自动降级，不影响推理主Pipeline运行。


---

该README为 Draft 版本，会随着模块化与测试完善持续更新，欢迎建议！。
