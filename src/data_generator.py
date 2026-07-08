import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def generate_channel_data(
    data_size=config.DATA_SIZE,
    output_path=config.DATA_PATH,
    random_seed=config.RANDOM_SEED,
    plot_path=config.CHANNEL_SIMULATION_PLOT,
    show_plot=True,
):
    """Generate simulated SINR data and save it as a CSV file."""
    rng = np.random.default_rng(random_seed)

    # 1. 生成时间点
    t = np.arange(data_size)

    # 2. 基础下降趋势 (路径损耗)
    path_loss = 30 - 0.008 * t

    # 3. 增加“慢衰落” (阴影衰落 Shadowing)
    shadowing = 3 * np.sin(2 * np.pi * 0.002 * t) + 1.5 * np.cos(2 * np.pi * 0.005 * t)

    # 4. 增加“突发事件” (Sudden Event)
    sudden_drop = np.zeros(data_size)
    sudden_drop[800:min(1000, data_size)] = -5

    # 5. 增加“快衰落” (瑞利衰落/多径效应 Fast Fading)
    fast_fading = rng.normal(0, 1.2, data_size)

    # 6. 最终信号质量
    sinr = path_loss + shadowing + sudden_drop + fast_fading

    df = pd.DataFrame({
        'timestamp': t,
        'sinr': sinr,
    })
    df.to_csv(output_path, index=False)
    print(f"Success: Data generated and saved to {output_path}")

    if show_plot or plot_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, sinr, label='Combined Signal (PathLoss + Shadowing + Fading)', alpha=0.75, linewidth=1.6)
        ax.axvspan(800, min(1000, data_size), color='red', alpha=0.12, label='Sudden Degradation Zone')
        ax.set_title("Multi-component Wireless Channel Simulation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal Quality (SINR in dB)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.45)
        fig.tight_layout()
        if plot_path:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180)
            print(f"Success: Simulation plot saved to {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return df


if __name__ == "__main__":
    generate_channel_data()
