import requests
import time
import base64


class TAPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def image_to_base64(self, image_path):
        """将图像转换为base64格式"""
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    def create_t2v_task(
        self,
        prompt,
        resolution="720p",
        seed=42,
        negative_prompt="",
        aspect_ratio="16:9",
    ):
        """发起t2v任务"""
        payload = {
            "prompt": prompt,
            "seed": seed,
            "target_video_resolution": resolution,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/", json=payload)
        return response.json()

    def create_i2v_task(
        self,
        prompt,
        first_image_path,
        resolution="720p",
        seed=42,
        negative_prompt="",
    ):
        """发起i2v任务"""
        # 支持多种图像输入格式
        if first_image_path.startswith("http"):
            # URL格式
            image_input = first_image_path
        else:
            # 本地文件转base64
            image_input = self.image_to_base64(first_image_path)

        payload = {
            "prompt": prompt,
            "first_image_path": image_input,
            "seed": seed,
            "target_video_resolution": resolution,
            "negative_prompt": negative_prompt,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/", json=payload)
        return response.json()

    def get_task_status(self, task_id):
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/v1/tasks/{task_id}/status")
        return response.json()

    def wait_for_completion(self, task_id, timeout=300):
        """等待任务完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            if status.get("status") == "completed":
                return True
            elif status.get("status") == "failed":
                return False
            time.sleep(2)
        return False

    def download_result(self, task_id, output_path):
        """下载结果视频"""
        response = requests.get(
            f"{self.base_url}/v1/tasks/{task_id}/result", stream=True
        )
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
