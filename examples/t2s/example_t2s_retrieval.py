import time
import torch
import numpy as np
import random
from pathlib import Path

# simlulate the 24G vram on h100, remove it on 4090 or A10
torch.cuda.set_per_process_memory_fraction(0.3, 0)
from teleai_pipe.pipelines.teleai_t2s_retrieval import (
    TeleAIT2SRetrivalPipeline,
)
from motion_visualizer import save_motion_gif, save_pose_image


def set_seed_anything(seed: int = 42):
    """
    设置所有常用库的随机种子，确保实验结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

    torch.backends.cudnn.deterministic = True  # 避免不确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用优化（为了可复现性）

    # print(f"✅ Random seed set to {seed}")


def interpolate(motion, n):
    N, _, _ = motion.shape
    inter_frames = []
    for i in range(N - 1):
        inter_frames.append(motion[i])
        cur_frame = motion[i]
        next_frame = motion[i + 1]
        for j in range(1, n + 1):
            alpha = j / (n + 1)
            inter_frame = cur_frame * (1 - alpha) + alpha * next_frame
            inter_frames.append(inter_frame)
    inter_frames.append(motion[-1])
    inter_motion = np.array(inter_frames)
    inter_motion = torch.tensor(inter_motion)
    return inter_motion


def get_dummy_control_pose():
    # 中心点
    cx, cy = 960, 540
    # 构建骨骼点（模拟一个跳舞动作的首帧）
    joints = np.array(
        [
            [cx, cy],  # 0: pelvis
            [cx + 30, cy + 40],  # 1: right hip
            [cx - 30, cy + 40],  # 2: left hip
            [cx, cy - 40],  # 3: lower spine
            [cx + 30, cy + 120],  # 4: right knee
            [cx - 30, cy + 120],  # 5: left knee
            [cx, cy - 80],  # 6: mid spine
            [cx + 30, cy + 200],  # 7: right ankle
            [cx - 30, cy + 200],  # 8: left ankle
            [cx, cy - 120],  # 9: upper spine
            [cx + 40, cy + 230],  # 10: right foot
            [cx - 40, cy + 230],  # 11: left foot
            [cx, cy - 160],  # 12: neck
            [cx + 20, cy - 160],  # 13: right clavicle
            [cx - 20, cy - 160],  # 14: left clavicle
            [cx, cy - 200],  # 15: head top
            [cx + 60, cy - 160],  # 16: right shoulder
            [cx - 60, cy - 160],  # 17: left shoulder
            [cx + 100, cy - 160],  # 18: right elbow
            [cx - 100, cy - 160],  # 19: left elbow
            [cx + 140, cy - 160],  # 20: right wrist
            [cx - 140, cy - 160],  # 21: left wrist
            [cx + 160, cy - 160],  # 22: right hand
            [cx - 160, cy - 160],  # 23: left hand
        ],
        dtype=np.float32,
    )
    # 归一化到 [0, 1] 区间（x:1920, y:1080）
    joints_normalized = joints.copy()
    joints_normalized[:, 0] /= 1920.0
    joints_normalized[:, 1] /= 1080.0

    # 转为 PyTorch tensor
    control = torch.tensor(joints_normalized, dtype=torch.float32).to("cuda")
    return control


if __name__ == "__main__":
    set_seed_anything(42)

    vae_path = (
        "/nvfile-heatstorage/jchen/TeleAI-Pipe/t2s_v0.0.1/vae_epoch_22_loss_0.0033.ckpt"
    )
    tf_path = (
        "/nvfile-heatstorage/jchen/TeleAI-Pipe/t2s_v0.0.1/tf_epoch_103_acc_0.3474.ckpt"
    )
    text_dir = "/nvfile-heatstorage/hzh/motion/texts"
    motion_dir = "/nvfile-heatstorage/hzh/motion/poses"
    model_dir = "/nvfile-heatstorage/model_zoo/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    pipe = TeleAIT2SRetrivalPipeline(vae_path, tf_path, text_dir, motion_dir, model_dir)

    # Get script from text
    infer_opt = {
        "timesteps": 10,
        "cond_scale": 2,
        "temperature": 1.0,
        "topk_filter_thres": 0.9,
        "gsample": False,
        "force_mask": False,
    }
    control = get_dummy_control_pose()
    motion = pipe(
        "A dance performance of a water-skating figure skater, standing upright, extending the body, bulging chest, person breathing deeply, body forwarding, person jumping with legs outstretched, arms up, body rotating in the air, maintaining posture, person tightening the body, legs together, when entering the water, hands straight"
        "跳水运动员完美演绎水上跃入技巧，人站立，伸展，挺胸，人深呼吸，身体前倾，人弯腿起跳，双臂向上伸展，身体在空中旋转，保持姿态，人身体紧绷，双腿并拢，入水时手臂伸直",
        control,
        100,
        infer_opt,
    )

    save_motion_gif(motion, "output.gif")
