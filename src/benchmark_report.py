import json

import matplotlib.pyplot as plt
import numpy as np

import config
from data_generator import generate_channel_data
from dl_model_training import train_lstm
from early_warning import run_warning_engine
from model_training import train_random_forest


def save_model_comparison(rf_result, lstm_result, plot_path=config.MODEL_COMPARISON_PLOT):
    """Save a compact model comparison chart for the README."""
    labels = ["Random Forest", "LSTM"]
    mae_values = [rf_result["mae"], lstm_result["mae"]]
    rmse_values = [rf_result["rmse"], lstm_result["rmse"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mae_bars = ax.bar(x - width / 2, mae_values, width, label="MAE", color="#2563eb")
    rmse_bars = ax.bar(x + width / 2, rmse_values, width, label="RMSE", color="#f97316")

    ax.set_title("Model Benchmark on Simulated SINR Test Split")
    ax.set_ylabel("Error (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bars in (mae_bars, rmse_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Success: Model comparison plot saved to {plot_path}")


def save_metrics(rf_result, lstm_result, warning_result, output_path=config.BENCHMARK_METRICS_PATH):
    """Save benchmark metrics used by the README."""
    metrics = {
        "dataset": {
            "data_size": config.DATA_SIZE,
            "window_size": config.WINDOW_SIZE,
            "test_size": config.TEST_SIZE,
            "random_seed": config.RANDOM_SEED,
        },
        "random_forest": {
            "mae_db": round(float(rf_result["mae"]), 4),
            "rmse_db": round(float(rf_result["rmse"]), 4),
            "n_estimators": int(rf_result["n_estimators"]),
        },
        "lstm": {
            "mae_db": round(float(lstm_result["mae"]), 4),
            "rmse_db": round(float(lstm_result["rmse"]), 4),
            "epochs": config.LSTM_EPOCHS,
            "hidden_size": config.LSTM_HIDDEN_SIZE,
        },
        "warning_engine": {
            "critical_count": len(warning_result["critical_warnings"]),
            "buffer_count": len(warning_result["buffer_warnings"]),
            "start_point": warning_result["start_point"],
            "end_point": warning_result["end_point"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Success: Benchmark metrics saved to {output_path}")
    return metrics


def run_benchmark_report(show_plots=False):
    """Regenerate data, models, plots, and README benchmark metrics."""
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    generate_channel_data(show_plot=show_plots)
    rf_result = train_random_forest(show_plot=show_plots)
    lstm_result = train_lstm(show_plot=show_plots)
    warning_result = run_warning_engine(show_plot=show_plots, verbose=False)

    save_model_comparison(rf_result, lstm_result)
    metrics = save_metrics(rf_result, lstm_result, warning_result)

    print("\nBenchmark summary")
    print(f"- Random Forest: MAE {metrics['random_forest']['mae_db']:.2f} dB, RMSE {metrics['random_forest']['rmse_db']:.2f} dB")
    print(f"- LSTM: MAE {metrics['lstm']['mae_db']:.2f} dB, RMSE {metrics['lstm']['rmse_db']:.2f} dB")
    print(
        "- Warning engine: "
        f"{metrics['warning_engine']['critical_count']} critical, "
        f"{metrics['warning_engine']['buffer_count']} buffer alerts"
    )
    return metrics


if __name__ == "__main__":
    run_benchmark_report()
