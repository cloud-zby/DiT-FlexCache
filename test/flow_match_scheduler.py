import matplotlib.pyplot as plt
from teleai_pipe.schedulers.flow_match import FlowMatchScheduler


def plot_shift_comparison():
    shifts = [0.5, 1.0, 3.0, 5.0, 16.0]
    colors = ["blue", "green", "red", "purple", "orange"]
    num_steps = 100

    plt.figure(figsize=(12, 6))

    # 绘制sigma曲线
    plt.subplot(1, 2, 1)
    for shift, color in zip(shifts, colors):
        scheduler = FlowMatchScheduler(
            num_inference_steps=num_steps,
            shift=shift,
            reverse_sigmas=False,
            inverse_timesteps=False,
        )
        plt.plot(
            scheduler.sigmas.numpy(), label=f"shift={shift}", color=color, linewidth=2
        )
    plt.title("Sigma Schedule Comparison")
    plt.xlabel("Step")
    plt.ylabel("Sigma Value")
    plt.legend()

    # 绘制step变化曲线
    plt.subplot(1, 2, 2)
    for shift, color in zip(shifts, colors):
        scheduler = FlowMatchScheduler(
            num_inference_steps=num_steps,
            shift=shift,
            reverse_sigmas=False,
            inverse_timesteps=False,
        )
        steps = scheduler.sigmas.diff().abs().numpy()  # 计算相邻sigma差值
        plt.plot(steps, label=f"shift={shift}", color=color, linestyle="--", alpha=0.7)
    plt.title("Step Size Comparison")
    plt.xlabel("Step Index")
    plt.ylabel("Delta Sigma")
    plt.legend()

    plt.tight_layout()
    plt.savefig("shift_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_shift_comparison()
