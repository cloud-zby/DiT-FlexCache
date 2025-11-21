from teleai_pipe.server.client import TAPClient

# 使用示例
if __name__ == "__main__":
    client = TAPClient("http://10.127.16.3:25584")

    # 1. 发起任务
    result = client.create_i2v_task(
        prompt="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
        first_image_path="/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/examples/data/101235-video-720_0.png",
    )

    task_id = result.get("task_id")
    print(f"Task created: {task_id}")

    # 2. 等待完成
    if client.wait_for_completion(task_id):
        print("Task completed successfully")

        # 3. 下载结果
        video_path = "server_video.mp4"
        if client.download_result(task_id, video_path):
            print(f"Video downloaded successfully to {video_path}")
    else:
        print("Task failed or timed out")
