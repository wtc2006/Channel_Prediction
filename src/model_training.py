import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import config


def create_window_features(data, window_size=config.WINDOW_SIZE):
    """Create sliding-window features with a one-step trend feature."""
    if len(data) <= window_size:
        raise ValueError("Data length must be greater than WINDOW_SIZE.")

    X = []
    y = []

    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        trend = window[-1] - window[-2]
        features = np.append(window, trend)

        X.append(features)
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def train_random_forest(
    data_path=config.DATA_PATH,
    model_path=config.MODEL_PATH,
    window_size=config.WINDOW_SIZE,
    test_size=config.TEST_SIZE,
    random_seed=config.RANDOM_SEED,
    n_estimators=config.RF_ESTIMATORS,
    plot_path=config.RF_RESULTS_PLOT,
    show_plot=True,
):
    """Train the Random Forest baseline and save the model."""
    df = pd.read_csv(data_path)
    data = df['sinr'].values
    X, y = create_window_features(data, window_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("Success: Model optimized!")
    print("--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} dB")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f} dB")

    joblib.dump(model, model_path)
    print(f"Success: Model saved to {model_path}")

    if show_plot or plot_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test[:150], label='Ground Truth (Actual)', color='#2563eb', linewidth=1.8)
        ax.plot(y_pred_test[:150], label='Random Forest Prediction', color='#dc2626', linestyle='--', linewidth=1.8)
        ax.set_title(f"Random Forest Channel Prediction | MAE {mae:.2f} dB, RMSE {rmse:.2f} dB")
        ax.set_xlabel("Test sample")
        ax.set_ylabel("SINR (dB)")
        ax.legend()
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        if plot_path:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180)
            print(f"Success: RF result plot saved to {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'n_estimators': n_estimators,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
    }


if __name__ == "__main__":
    train_random_forest()
