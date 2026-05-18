import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

# 1. 加载模型和数据
model = joblib.load(config.MODEL_PATH)
df = pd.read_csv(config.DATA_PATH)
data = df['sinr'].values

# 2. 预警参数
CRITICAL_THRESHOLD = config.CRITICAL_THRESHOLD
WARNING_BUFFER = config.WARNING_BUFFER
window_size = config.WINDOW_SIZE

# 3. 模拟实时预警流程
results = []
warnings_critical = []
warnings_buffer = []

# 我们从第 750 个点开始，正好覆盖“突发劣化区” (800-1000)
start_point = 750
end_point = 1100

for i in range(start_point, end_point):
    # 提取当前窗口并增加“趋势特征” (必须与训练一致)
    window = data[i - window_size : i]
    trend = window[-1] - window[-2]
    features = np.append(window, trend).reshape(1, -1)
    
    # 预测
    prediction = model.predict(features)[0]
    results.append(prediction)
    
    # 预警逻辑
    # TODO: 用户任务 - 根据预测值 prediction 判断是否触发预警
    # 请在这里实现逻辑：
    # 1. 如果 prediction 小于 CRITICAL_THRESHOLD，判定为严重预警，记录到 warnings_critical
    # 2. 否则如果 prediction 小于 (CRITICAL_THRESHOLD + WARNING_BUFFER)，判定为普通预警，记录到 warnings_buffer
    if prediction < CRITICAL_THRESHOLD:
        warnings_critical.append(i)
        print(f"!!! CRITICAL WARNING: Time {i}, Predicted {prediction:.2f}dB (Below Threshold!)")       
    elif prediction < (CRITICAL_THRESHOLD + WARNING_BUFFER):
        warnings_buffer.append(i)
        print(f"Warning: Time {i}, Predicted {prediction:.2f}dB (Entering Buffer Zone)")
    

# 4. 更加专业的图表展示
plt.figure(figsize=(14, 7))

# 真实信号与 AI 预测
time_range = range(start_point, end_point)
plt.plot(time_range, data[start_point:end_point], label='Actual Signal', color='blue', alpha=0.6)
plt.plot(time_range, results, label='AI Future Prediction', color='darkorange', linestyle='--')

# 绘制警戒区域
plt.axhline(y=CRITICAL_THRESHOLD, color='red', linestyle='-', linewidth=2, label='Critical Threshold (15dB)')
plt.axhline(y=CRITICAL_THRESHOLD + WARNING_BUFFER, color='yellow', linestyle='--', label='Warning Buffer (17dB)')
plt.fill_between(time_range, CRITICAL_THRESHOLD, CRITICAL_THRESHOLD + WARNING_BUFFER, color='yellow', alpha=0.1, label='Buffer Zone')
plt.fill_between(time_range, 0, CRITICAL_THRESHOLD, color='red', alpha=0.1, label='Danger Zone')

# 标记预警点
if warnings_critical:
    plt.scatter(warnings_critical, [CRITICAL_THRESHOLD]*len(warnings_critical), color='red', marker='x', s=50, label='Critical Alarm')

plt.title("Advanced Early Warning System: Trend Prediction & Zone Monitoring")
plt.xlabel("Time (s)")
plt.ylabel("Signal Quality (SINR in dB)")
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.ylim(min(data[start_point:end_point])-2, max(data[start_point:end_point])+2)
plt.show()
