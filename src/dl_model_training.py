import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
import joblib

# 1. 准备数据
df = pd.read_csv(config.DATA_PATH)
data = df['sinr'].values

X, y = [], []
for i in range(len(data) - config.WINDOW_SIZE):
    X.append(data[i : i + config.WINDOW_SIZE])
    y.append(data[i + config.WINDOW_SIZE])

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集 (保持与随机森林一致)
split = int(len(X) * (1 - config.TEST_SIZE))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 转换为 PyTorch 张量
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)

# 2. 定义 LSTM 模型架构
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 3. 初始化模型、损失函数和优化器
model = SimpleLSTM()
criterion = nn.MSELoss() # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam 优化器

# 4. 训练循环
epochs = 100
losses = []
print("Starting LSTM Training...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).numpy()
    mae = np.mean(np.abs(predictions - y_test_t.numpy()))
    print(f"\nLSTM Success: Training Completed!")
    print(f"LSTM Mean Absolute Error (MAE): {mae:.4f} dB")

# 6. 保存模型 (注意：PyTorch 通常保存 state_dict)
torch.save(model.state_dict(), 'channel_lstm_model.pth')
print("Success: LSTM model saved to channel_lstm_model.pth")

# 7. 可视化
plt.figure(figsize=(12, 6))
plt.plot(y_test[:150], label='Ground Truth', color='blue')
plt.plot(predictions[:150], label='LSTM Prediction', color='green', linestyle='--')
plt.title("Deep Learning: LSTM Channel Prediction Performance")
plt.legend()
plt.grid(True)
plt.show()
