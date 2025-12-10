# 安装
## 系统配置
设置代理
```bash
bash /gemini/space/zby/tools/setup_env.sh
```
安装虚拟环境库
```bash
apt install python3.10-venv
```
## 安装telepipe
打开链接，下载代码，使用http下载
https://www.srdcloud.cn/code/50206/repoView/339817/codeFile/tree?key=master
密码不是研发云的密码，是研发云代码仓库页面上“HTTPS密码”点开可以看到
```bash
git clone https://zhoubingye@code.srdcloud.cn/P24HQASYF0004/AI-Infra/teleai_efficient
cd teleai_efficient && git checkout feat/cache
```
创建虚拟环境
```bash
python -m venv dit-cache
source dit-cache/bin/activate
source /gemini/space/zby/repos/teleai_efficient/dit-cache/bin/activate
```
安装
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install click uvicorn fastapi python-multipart
pip install -v -e .
```
## 安装telefuser
从共享存储`/gemini/space/zby/repos/teleai_efficient/TeleFuser`进行`cp -r`拷贝
```bash
pip install -e .
```
## 安装fluxon部署仓库
https://www.srdcloud.cn/code/50206/repoView/341212/codeFile/tree?key=main
```bash
git clone https://zhoubingye@code.srdcloud.cn/P24HQASYF0004/AI-Infra/fluxon_deploy
cd fluxon_deploy
```
切换分支
```bash
git checkout main
```
安装wheel
```bash
pip install fluxon_kv-0.1.1-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
## 安装fluxon代码仓库
https://www.srdcloud.cn/code/50206/repoView/283696/codeFile/tree?key=github_master
```bash
git clone https://zhoubingye@code.srdcloud.cn/P24HQASYF0004/AI-Infra/py-kvcache-api-layer
cd py-kvcache-api-layer
```
切换分支
```bash
git checkout github_master
```
安装fluxon项目
```bash
pip install -e .
```
最后，同一个目录下应该有TeleFuser  fluxon_deploy  teleai_efficient这三个文件夹，并在fluxon_deploy目录下存放fluxon_deploy/py-kvcache-api-layer
## 启动KVCache后端
先修改下面这个文件里面的配置内容，关键配置：
```python
ETCD = "10.127.29.225:32379"
"monitoring": {"prom_remote_write_url": "http://10.127.29.225:34030/v1/prometheus/write"},
CLUSTER_NAME = "fluxon_dit_cluster"
SHARED_MEM = "/gemini/space/zby/repos/fluxon_deploy/py-kvcache-api-layer/kvcache_api_layer/dit_share_mem"
```
另外，`CLUSTER_NAME`和`SHARED_MEM`需要和python用户代码里面的设置一样
生成config
```bash
python /gemini/space/zby/repos/fluxon_deploy/py-kvcache-api-layer/kvcache_api_layer/examples/fluxon_allinone/gen_config.py
```
启动master
```bash
source /gemini/space/zby/repos/teleai_efficient/dit-cache/bin/activate
python /gemini/space/zby/repos/fluxon_deploy/py-kvcache-api-layer/kvcache_api_layer/examples/fluxon_allinone/scripts/start_fluxonkv_master.py
```
启动kvclient
```bash
source /gemini/space/zby/repos/teleai_efficient/dit-cache/bin/activate
python /gemini/space/zby/repos/fluxon_deploy/py-kvcache-api-layer/kvcache_api_layer/examples/fluxon_allinone/scripts/start_kvclient.py
```
## 运行
设置以下环境变量：
```bash
export FLEXCACHE_KV_CONFIG=examples/flexcache/KVstore.yaml
export QDRANT_URL=http://127.0.0.1:6333
export TELEAIPIPE_CACHE_DIR=./flexcache
```
先下载qdrant-x86_64-unknown-linux-musl.tar.gz，然后启动
```bash
cd examples/flexcache && ./qdrant
```
启动服务器
```bash
cd /gemini/space/zby/repos/teleai_efficient
python teleai_pipe/api_server.py -pp examples/flexcache/wan22_14b_text_to_video_prompt_enhance_h100.py --task t2v --host 0.0.0.0 --port 8000 -g 2
```
发送请求
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
或者直接运行
```bash
bash examples/teleai_video_v1_3/wan22_14b_text_to_video_prompt_enhance_a800.py
```




