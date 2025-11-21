import base64
import io
import signal
import sys

import psutil
import torch
from PIL import Image
from loguru import logger
from pydantic import BaseModel
import random
import string
import time
from datetime import datetime
import importlib


def import_function_from_file(file_path, function_name):
    """
    从指定路径的Python文件中导入特定函数

    参数:
        file_path: .py文件的完整路径
        function_name: 要导入的函数名

    返回:
        导入的函数对象
    """
    # 生成模块名(基于文件名)
    module_name = file_path.split("/")[-1].split(".")[0]

    # 加载模块规范
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # 创建新模块
    module = importlib.util.module_from_spec(spec)

    # 将模块添加到sys.modules中
    sys.modules[module_name] = module

    # 执行模块
    spec.loader.exec_module(module)

    # 获取并返回函数
    return getattr(module, function_name)


def generate_task_id():
    """
    Generate a random task ID in the format XXXX-XXXX-XXXX-XXXX-XXXX.
    Features:
    1. Does not modify the global random state.
    2. Each X is an uppercase letter or digit (0-9).
    3. Combines time factors to ensure high randomness.
    """
    # Save the current random state (does not affect external randomness)
    original_state = random.getstate()

    try:
        # Define character set (uppercase letters + digits)
        characters = string.ascii_uppercase + string.digits

        # Create an independent random instance
        local_random = random.Random(time.perf_counter_ns())

        # Generate 5 groups of 4-character random strings
        groups = []
        for _ in range(5):
            # Mix new time factor for each group
            time_mix = int(datetime.now().timestamp())
            local_random.seed(
                time_mix + local_random.getstate()[1][0] + time.perf_counter_ns()
            )

            groups.append("".join(local_random.choices(characters, k=4)))

        return "-".join(groups)

    finally:
        # Restore the original random state
        random.setstate(original_state)


class ProcessManager:
    @staticmethod
    def kill_all_related_processes():
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception as e:
                logger.info(f"Failed to kill child process {child.pid}: {e}")
        try:
            current_process.kill()
        except Exception as e:
            logger.info(f"Failed to kill main process: {e}")

    @staticmethod
    def signal_handler(sig, frame):
        logger.info("\nReceived Ctrl+C, shutting down all related processes...")
        ProcessManager.kill_all_related_processes()
        sys.exit(0)

    @staticmethod
    def register_signal_handler():
        signal.signal(signal.SIGINT, ProcessManager.signal_handler)


class TaskStatusMessage(BaseModel):
    task_id: str


class TensorTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def prepare_tensor(self, data) -> str:
        self.buffer.seek(0)
        self.buffer.truncate()
        torch.save(self.to_device(data, "cpu"), self.buffer)
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_tensor(self, tensor_base64: str, device="cuda"):
        tensor_bytes = base64.b64decode(tensor_base64)
        with io.BytesIO(tensor_bytes) as buffer:
            return self.to_device(torch.load(buffer), device)


class ImageTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def prepare_image(self, image: Image.Image):
        self.buffer.seek(0)
        self.buffer.truncate()
        image.save(self.buffer, format="PNG")
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_image(self, image_base64: bytes) -> Image.Image:
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as buffer:
            return Image.open(buffer).convert("RGB")
