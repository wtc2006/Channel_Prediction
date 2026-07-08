"""
Configuration module for the Channel Prediction & Early Warning System.
All hyperparameters and file paths are centralized here for easy maintenance.
"""

from pathlib import Path

# 核心配置参数
WINDOW_SIZE = 15          # 滑动窗口大小
DATA_SIZE = 1200          # 生成数据点数量
TEST_SIZE = 0.2           # 测试集比例
RANDOM_SEED = 42          # 随机种子，保证结果可复现

# 预警阈值
CRITICAL_THRESHOLD = 15   # 严重危险线 (dB)
WARNING_BUFFER = 2        # 预警缓冲带 (dB)

# LSTM 配置
LSTM_EPOCHS = 100
LSTM_HIDDEN_SIZE = 32
LSTM_LEARNING_RATE = 0.01

# 文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'channel_data.csv'
MODEL_PATH = PROJECT_ROOT / 'channel_model.pkl'
LSTM_MODEL_PATH = PROJECT_ROOT / 'channel_lstm_model.pth'
