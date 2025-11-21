# ✨ [新建] tests/test_video_api.py
import requests
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_IMAGE = Path(__file__).parent.parent / "data" / "wan_input.jpg"


def test_video_generation():
    # 准备测试数据
    data = {}
    print(data)

    # 发送请求
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/generate_video", json=data)

    # 验证结果
    print(response)
    assert response.status_code == 200
    assert "video/mp4" in response.headers["Content-Type"]

    # 保存测试结果
    output_path = "test_output.mp4"
    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Test passed! Video saved to {output_path}")
    print(f"Generation time: {time.time()-start_time:.2f}s")


if __name__ == "__main__":
    test_video_generation()
