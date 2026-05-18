import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import joblib
import matplotlib.pyplot as plt
import config

# 1. 读取数据
df = pd.read_csv(config.DATA_PATH)
data = df['sinr'].values

# 2. 准备滑动窗口数据 (X, y)
X = []
y = []
window_size = config.WINDOW_SIZE

for i in range(len(data) - window_size):
    window = data[i : i + window_size]
    trend = window[-1] - window[-2] 
    features = np.append(window, trend) 
    
    X.append(features)
    y.append(data[i + window_size])

X = np.array(X)
y = np.array(y)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, shuffle=False)

# 4. 创建并训练模型 (随机森林)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 评估模型
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Success: Model optimized!")
print(f"--- Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.4f} dB")
print(f"Root Mean Square Error (RMSE): {rmse:.4f} dB")

# 6. 保存模型
joblib.dump(model, config.MODEL_PATH)
print(f"Success: Model saved to {config.MODEL_PATH}")

# 7. 可视化测试集结果
plt.figure(figsize=(12, 6))
plt.plot(y_test[:150], label='Ground Truth (Actual)', color='blue')
plt.plot(y_pred_test[:150], label='Optimized Prediction (RandomForest)', color='red', linestyle='--')
plt.title("Model Optimization: Training/Testing Performance")
plt.legend()
plt.grid(True)
plt.show()
