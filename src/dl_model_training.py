import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import config


class SimpleLSTM(nn.Module):
    """A compact LSTM regressor for one-step SINR prediction."""

    def __init__(self, input_size=1, hidden_size=config.LSTM_HIDDEN_SIZE, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_lstm_sequences(data, window_size=config.WINDOW_SIZE):
    """Create sliding-window sequences for LSTM training."""
    if len(data) <= window_size:
        raise ValueError("Data length must be greater than WINDOW_SIZE.")

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def train_lstm(
    data_path=config.DATA_PATH,
    model_path=config.LSTM_MODEL_PATH,
    window_size=config.WINDOW_SIZE,
    test_size=config.TEST_SIZE,
    epochs=config.LSTM_EPOCHS,
    random_seed=config.RANDOM_SEED,
    plot_path=config.LSTM_RESULTS_PLOT,
    show_plot=True,
):
    """Train the LSTM model and save a checkpoint with normalization metadata."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    df = pd.read_csv(data_path)
    data = df['sinr'].values
    X, y = create_lstm_sequences(data, window_size)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_mean = float(X_train.mean())
    train_std = float(X_train.std())
    if train_std == 0:
        raise ValueError("Training data standard deviation is zero; cannot normalize.")

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    y_train_norm = (y_train - train_mean) / train_std

    X_train_t = torch.FloatTensor(X_train_norm).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train_norm).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test_norm).unsqueeze(-1)

    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LEARNING_RATE)
    losses = []

    print("Starting LSTM Training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions_norm = model(X_test_t).numpy().reshape(-1)
        predictions = predictions_norm * train_std + train_mean
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nLSTM Success: Training Completed!")
    print(f"LSTM Mean Absolute Error (MAE): {mae:.4f} dB")
    print(f"LSTM Root Mean Square Error (RMSE): {rmse:.4f} dB")

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'window_size': window_size,
            'hidden_size': config.LSTM_HIDDEN_SIZE,
            'train_mean': train_mean,
            'train_std': train_std,
            'epochs': epochs,
            'mae': mae,
            'rmse': rmse,
        },
        model_path,
    )
    print(f"Success: LSTM model saved to {model_path}")

    if show_plot or plot_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        axes[0].plot(y_test[:150], label='Ground Truth', color='#2563eb', linewidth=1.8)
        axes[0].plot(predictions[:150], label='LSTM Prediction', color='#16a34a', linestyle='--', linewidth=1.8)
        axes[0].set_title(f"LSTM Prediction | MAE {mae:.2f} dB, RMSE {rmse:.2f} dB")
        axes[0].set_xlabel("Test sample")
        axes[0].set_ylabel("SINR (dB)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.35)

        axes[1].plot(losses, color='#7c3aed', linewidth=1.8)
        axes[1].set_title("Normalized Training Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE")
        axes[1].grid(True, alpha=0.35)

        fig.tight_layout()
        if plot_path:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180)
            print(f"Success: LSTM result plot saved to {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'losses': losses,
        'y_test': y_test,
        'predictions': predictions,
        'train_mean': train_mean,
        'train_std': train_std,
    }


if __name__ == "__main__":
    train_lstm()
