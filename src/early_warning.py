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
            print(f"!!! CRITICAL WARNING: Time {i}, Predicted {prediction:.2f}dB (Below Threshold!)")
        elif level == 'warning':
            warnings_buffer.append(i)
            print(f"Warning: Time {i}, Predicted {prediction:.2f}dB (Entering Buffer Zone)")

    if show_plot:
        time_range = range(start_point, end_point)

        plt.figure(figsize=(14, 7))
        plt.plot(time_range, data[start_point:end_point], label='Actual Signal', color='blue', alpha=0.6)
        plt.plot(time_range, results, label='AI Future Prediction', color='darkorange', linestyle='--')

        plt.axhline(y=critical_threshold, color='red', linestyle='-', linewidth=2, label='Critical Threshold (15dB)')
        plt.axhline(
            y=critical_threshold + warning_buffer,
            color='yellow',
            linestyle='--',
            label='Warning Buffer (17dB)',
        )
        plt.fill_between(
            time_range,
            critical_threshold,
            critical_threshold + warning_buffer,
            color='yellow',
            alpha=0.1,
            label='Buffer Zone',
        )
        plt.fill_between(time_range, 0, critical_threshold, color='red', alpha=0.1, label='Danger Zone')

        if warnings_critical:
            plt.scatter(
                warnings_critical,
                [critical_threshold] * len(warnings_critical),
                color='red',
                marker='x',
                s=50,
                label='Critical Alarm',
            )
        if warnings_buffer:
            plt.scatter(
                warnings_buffer,
                [critical_threshold + warning_buffer] * len(warnings_buffer),
                color='gold',
                marker='o',
                s=35,
                label='Warning Alarm',
            )

        plt.title("Advanced Early Warning System: Trend Prediction & Zone Monitoring")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Quality (SINR in dB)")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.ylim(min(data[start_point:end_point]) - 2, max(data[start_point:end_point]) + 2)
        plt.show()

    return {
        'predictions': np.array(results),
        'critical_warnings': warnings_critical,
        'buffer_warnings': warnings_buffer,
    }


if __name__ == "__main__":
    run_warning_engine()
