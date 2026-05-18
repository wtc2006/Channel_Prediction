import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import config

# 设置随机种子，保证每次运行结果一致
np.random.seed(42)

# 1. 生成时间点
t = np.arange(config.DATA_SIZE) 

# 2. 基础下降趋势 (路径损耗)
path_loss = 30 - 0.008 * t 

# 3. 增加“慢衰落” (阴影衰落 Shadowing)
shadowing = 3 * np.sin(2 * np.pi * 0.002 * t) + 1.5 * np.cos(2 * np.pi * 0.005 * t)

# 4. 增加“突发事件” (Sudden Event)
sudden_drop = np.zeros(config.DATA_SIZE)
sudden_drop[800:1000] = -5 

# 5. 增加“快衰落” (瑞利衰落/多径效应 Fast Fading)
fast_fading = np.random.normal(0, 1.2, config.DATA_SIZE) 

# 6. 最终信号叠加
sinr = path_loss + shadowing + sudden_drop + fast_fading

# --- 数据持久化 ---
df = pd.DataFrame({
    'timestamp': t,
    'sinr': sinr
})

# 保存数据
df.to_csv(config.DATA_PATH, index=False)
print(f"Success: Data generated and saved to {config.DATA_PATH}")

# 7. 画图展示 (增加参考线和标注)
plt.figure(figsize=(12, 6))
plt.plot(t, sinr, label='Combined Signal (PathLoss + Shadowing + Fading)', alpha=0.7)
plt.axvspan(800, 1000, color='red', alpha=0.1, label='Sudden Degradation Zone') # 标注突发劣化区
plt.title("Optimized Data: Multi-component Wireless Channel Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Signal Quality (SINR in dB)")
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
