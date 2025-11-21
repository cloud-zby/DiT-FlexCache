# TeleImage_V7

## 启动方式
### Cilent

1. 首先运行指令,启动Qwen cilent

    ```bash
    vllm serve models/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.95 --uvicorn-log-level error --port 11451 --max_model_len 94720 --served-model-name  Qwen3-30B-A3B-Instruct-2507
    ```
    ```yaml

    # 如果修改上述启动命令，则需对应修改yaml文件配置
    cilent_base: "http://localhost:11451/v1" # 对应--port参数 
    served_model_name: Qwen3-30B-A3B-Instruct-2507 # 对应--served-model-name参数
    ```

2. 然后修改`main.py` 中的 `use_cilent`值为`True`,启动cilent模式。（自动改为读取config_cilent.yaml文件）


3. 运行
    ```bash
    cd codes
    python main.py # kill此程序不会影响vllm cilent程序
    ```

### 非客户端方式启动 
修改`main.py` 中的 `use_cilent`值为`False`.（读取config.yaml文件）
```bash
cd codes
python main.py
```

# TeleImage_V7.1 更新说明
1. 更新输入参数，包含highres，medres，lowres分别对应高中低清，每个模式下包含16:9，9:16，3:2，2:3，1:1五种尺寸选择
2. 更新使用方式，采用config.ymal表示纯python代码形式，采用config_client为使用API形式
3. 更新语义解析，防止语义扩写带来多余动作
4. 更新测试模式