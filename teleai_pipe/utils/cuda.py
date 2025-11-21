import pynvml


def filter_gpus_by_memory(min_free_memory=None, max_used_memory=None):
    """
    根据显存占用情况筛选符合条件的GPU ID

    参数:
        min_free_memory: 最小空闲显存(MB)，可选
        max_used_memory: 最大已用显存(MB)，可选

    返回:
        符合条件的GPU ID列表
    """
    try:
        # 初始化NVML
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        suitable_gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total = mem_info.total / (1024**2)  # 转换为MB
            used = mem_info.used / (1024**2)
            free = total - used
            # 检查是否满足条件
            condition_met = True

            if min_free_memory is not None and free < min_free_memory:
                condition_met = False

            if max_used_memory is not None and used > max_used_memory:
                condition_met = False

            if condition_met:
                suitable_gpus.append(i)

        return suitable_gpus

    except Exception as e:
        print(f"获取GPU信息时出错: {e}")
        return []
    finally:
        # 确保总是关闭NVML
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
