# Wireless Channel Prediction & Early Warning System 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

这是一个基于机器学习与深度学习的无线信道质量（SINR）预测与劣化预警系统。该项目模拟了真实的物理环境（路径损耗、阴影衰落、快衰落），并提供了从数据生成到智能预警的完整链路。

---

## 🌟 项目亮点 (Highlights)

- **多组件物理建模**：真实模拟无线信号的路径损耗、环境遮挡及多径效应。
- **混合算法架构**：内置 **Random Forest (随机森林)** 与 **LSTM (深度学习)** 双预测引擎。
- **工业级预警机制**：支持自适应缓冲区（Warning Buffer）的实时阈值监控。
- **架构化设计**：采用 `config.py` 统一管理参数，易于扩展与二次开发。

---

## 🛠️ 技术栈 (Tech Stack)

- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn (Random Forest)
- **深度学习**: PyTorch (LSTM)
- **可视化**: Matplotlib

---

## 🚀 快速开始 (Quick Start)

### 1. 环境安装
```bash
pip install numpy pandas matplotlib scikit-learn torch joblib
```

### 2. 一键运行全流程
```bash
python src/main.py
```

---

## 📂 目录结构 (Project Structure)

```text
.
├── src/
│   ├── config.py             # 系统核心参数配置
│   ├── data_generator.py     # 物理环境信号模拟器
│   ├── model_training.py     # 随机森林模型训练
│   ├── dl_model_training.py  # LSTM 深度学习模型训练
│   ├── early_warning.py      # 智能预警逻辑实现
│   └── main.py               # 项目总控入口
├── .gitignore                # Git 忽略文件配置
├── LICENSE                   # MIT 开源协议
├── requirements.txt          # 项目依赖清单
└── README.md                 # 项目说明文档
```

---

## 📈 可视化展示 (Visualizations)

### 1. 信道模拟 (Data Simulation)
真实模拟无线信号的路径损耗、环境遮挡及多径效应。
![Data Simulation](assets/01-channel-simulation.png)

### 2. 模型表现对比 (Model Performance)
对比基础机器学习与深度学习在处理复杂衰落时的预测能力。

| Random Forest (机器学习) | LSTM (深度学习) |
| :---: | :---: |
| ![RF Results](assets/02-ml-rf-results.png) | ![LSTM Results](assets/03-dl-lstm-results.png) |

### 3. 智能预警监控 (Smart Warning Monitoring)
动态展示信号进入“缓冲区”与“危险区”的过程，实现前瞻性预警。
![Warning System](assets/04-warning-monitoring.png)

---

## 🔬 核心原理 (How it Works)

1. **数据切片**: 利用滑动窗口（Sliding Window）将连续信号转化为模型可理解的特征矩阵。
2. **特征增强**: 在 [model_training.py](file:///e:/Channel_Prediction/first/model_training.py) 中引入了“趋势特征”，捕捉信号的一阶导数变化。
3. **深度学习**: [dl_model_training.py](file:///e:/Channel_Prediction/first/dl_model_training.py) 使用 LSTM 门控机制自动记忆长期的信道衰落规律。

---

## 🚀 未来计划 (Roadmap)

- [ ] 集成 Transformer 算法提升长序列预测能力。
- [ ] 开发基于 Flask/Streamlit 的 Web 实时监控看板。
- [ ] 支持多天线 (MIMO) 场景下的信道预测。

---

## 🤝 贡献与交流

欢迎提交 Issue 或 Pull Request 来优化模型！如果你觉得这个项目对你有帮助，请点一个 **Star** ⭐。
