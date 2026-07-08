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

    model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
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

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:150], label='Ground Truth (Actual)', color='blue')
        plt.plot(y_pred_test[:150], label='Optimized Prediction (RandomForest)', color='red', linestyle='--')
        plt.title("Model Optimization: Training/Testing Performance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
    }


if __name__ == "__main__":
    train_random_forest()
