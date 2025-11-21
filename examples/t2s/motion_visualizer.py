import matplotlib.pyplot as plt
from torch import Tensor
from matplotlib.animation import FuncAnimation, PillowWriter


LINKS = [
    [15, 12, 9, 6, 3, 0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [23, 21, 19, 17, 14, 9],
    [22, 20, 18, 16, 13, 9],
]


# 绘制一个motion，返回ani对象
def plot_motion(motion: Tensor, fps: int, resolution: tuple[int, int]) -> FuncAnimation:
    def update(frame):
        ax.clear()
        ax.set_xlabel(f"X ({frame})")
        ax.set_ylabel(f"Y ({frame})")
        ax.set_xlim(0, resolution[0])
        ax.set_ylim(0, resolution[1])
        x = motion[frame, :, 0]
        y = motion[frame, :, 1]
        ax.scatter(x, y)
        for link in LINKS:
            ax.plot(x[link], y[link], c="r")
        ax.invert_yaxis()
        return 1

    fig, ax = plt.subplots()
    if hasattr(motion, "numpy"):
        motion = motion.numpy()
    interval = 1000 / fps
    ani = FuncAnimation(
        fig, update, frames=motion.shape[0], interval=interval, blit=False
    )
    return ani


def save_pose_image(
    pose: Tensor,
    save_path: str,
    resolution: tuple[int, int] = (1, 1),
) -> None:

    if hasattr(pose, "numpy"):
        pose = pose.numpy()

    fig, ax = plt.subplots()
    ax.clear()
    ax.set_xlabel(f"X")
    ax.set_ylabel(f"Y")
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(0, resolution[1])
    x = pose[:, 0]
    y = pose[:, 1]
    ax.scatter(x, y)
    for link in LINKS:
        ax.plot(x[link], y[link], c="r")
    ax.invert_yaxis()
    plt.savefig(save_path)


# 在jupyter notebook中显示一个motion
# def show_motion(motion: Tensor, fps=30, resolution: tuple[int, int] = (1, 1)) -> None:
#     ani = plot_motion(motion, fps, resolution)
#     html = HTML(ani.to_jshtml())
#     # display(html)
#     plt.close()
#     return


# 将motion保存为gif
def save_motion_gif(
    motion: Tensor, save_path: str, fps=30, resolution: tuple[int, int] = (1920, 1080)
) -> None:
    ani = plot_motion(motion, fps, resolution)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    return
