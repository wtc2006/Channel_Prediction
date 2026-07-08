import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


def classify_prediction(
    prediction,
    critical_threshold=config.CRITICAL_THRESHOLD,
    warning_buffer=config.WARNING_BUFFER,
):
    """Classify a prediction into critical, warning, or normal."""
    if prediction < critical_threshold:
        return 'critical'
    if prediction < critical_threshold + warning_buffer:
        return 'warning'
    return 'normal'


def run_warning_engine(
    data_path=config.DATA_PATH,
    model_path=config.MODEL_PATH,
    start_point=750,
    end_point=1100,
    window_size=config.WINDOW_SIZE,
    critical_threshold=config.CRITICAL_THRESHOLD,
    warning_buffer=config.WARNING_BUFFER,
    plot_path=config.WARNING_MONITORING_PLOT,
    verbose=True,
    show_plot=True,
):
    """Run prediction-based early warning over a selected time range."""
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    data = df['sinr'].values

    if start_point < window_size:
        raise ValueError("start_point must be greater than or equal to WINDOW_SIZE.")

    end_point = min(end_point, len(data))
    results = []
    warnings_critical = []
    warnings_buffer = []

    for i in range(start_point, end_point):
        window = data[i - window_size:i]
        trend = window[-1] - window[-2]
        features = np.append(window, trend).reshape(1, -1)

        prediction = model.predict(features)[0]
        results.append(prediction)

        level = classify_prediction(prediction, critical_threshold, warning_buffer)
        if level == 'critical':
            warnings_critical.append(i)
            if verbose:
                print(f"!!! CRITICAL WARNING: Time {i}, Predicted {prediction:.2f}dB (Below Threshold!)")
        elif level == 'warning':
            warnings_buffer.append(i)
            if verbose:
                print(f"Warning: Time {i}, Predicted {prediction:.2f}dB (Entering Buffer Zone)")

    if show_plot or plot_path:
        time_range = range(start_point, end_point)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(time_range, data[start_point:end_point], label='Actual Signal', color='#2563eb', alpha=0.7, linewidth=1.8)
        ax.plot(time_range, results, label='AI Future Prediction', color='#f97316', linestyle='--', linewidth=1.8)

        ax.axhline(y=critical_threshold, color='#dc2626', linestyle='-', linewidth=2, label='Critical Threshold (15dB)')
        ax.axhline(
            y=critical_threshold + warning_buffer,
            color='#eab308',
            linestyle='--',
            label='Warning Buffer (17dB)',
        )
        ax.fill_between(
            time_range,
            critical_threshold,
            critical_threshold + warning_buffer,
            color='#fde047',
            alpha=0.18,
            label='Buffer Zone',
        )
        ax.fill_between(time_range, 0, critical_threshold, color='#fca5a5', alpha=0.16, label='Danger Zone')

        if warnings_critical:
            ax.scatter(
                warnings_critical,
                [critical_threshold] * len(warnings_critical),
                color='#dc2626',
                marker='x',
                s=50,
                label='Critical Alarm',
            )
        if warnings_buffer:
            ax.scatter(
                warnings_buffer,
                [critical_threshold + warning_buffer] * len(warnings_buffer),
                color='#ca8a04',
                marker='o',
                s=35,
                label='Warning Alarm',
            )

        ax.set_title("AI Early Warning: Trend Prediction & Risk Zone Monitoring")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal Quality (SINR in dB)")
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle=':', alpha=0.45)
        ax.set_ylim(min(data[start_point:end_point]) - 2, max(data[start_point:end_point]) + 2)
        fig.tight_layout()
        if plot_path:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180)
            print(f"Success: Warning plot saved to {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return {
        'predictions': np.array(results),
        'critical_warnings': warnings_critical,
        'buffer_warnings': warnings_buffer,
        'start_point': start_point,
        'end_point': end_point,
    }


if __name__ == "__main__":
    run_warning_engine()
