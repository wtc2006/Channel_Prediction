import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    show_plot=True,
):
    """Train the LSTM model and save its state_dict."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    df = pd.read_csv(data_path)
    data = df['sinr'].values
    X, y = create_lstm_sequences(data, window_size)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)

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
        predictions = model(X_test_t).numpy()
        mae = np.mean(np.abs(predictions - y_test_t.numpy()))

    print("\nLSTM Success: Training Completed!")
    print(f"LSTM Mean Absolute Error (MAE): {mae:.4f} dB")

    torch.save(model.state_dict(), model_path)
    print(f"Success: LSTM model saved to {model_path}")

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:150], label='Ground Truth', color='blue')
        plt.plot(predictions[:150], label='LSTM Prediction', color='green', linestyle='--')
        plt.title("Deep Learning: LSTM Channel Prediction Performance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'model': model,
        'mae': mae,
        'losses': losses,
        'y_test': y_test,
        'predictions': predictions,
    }


if __name__ == "__main__":
    train_lstm()
