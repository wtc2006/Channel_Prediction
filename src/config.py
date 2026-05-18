"""
Configuration module for the Channel Prediction & Early Warning System.
All hyperparameters and file paths are centralized here for easy maintenance.
"""

# 核心配置参数
WINDOW_SIZE = 15          # 滑动窗口大小
DATA_SIZE = 1200          # 生成数据点数量
TEST_SIZE = 0.2           # 测试集比例

# 预警阈值
CRITICAL_THRESHOLD = 15   # 严重危险线 (dB)
WARNING_BUFFER = 2        # 预警缓冲带 (dB)

# 文件路径
DATA_PATH = 'channel_data.csv'
MODEL_PATH = 'channel_model.pkl'
