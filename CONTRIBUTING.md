# Contributing

感谢你愿意改进这个项目。这个仓库主要关注无线信道 SINR 预测、模型对比和预警流程，欢迎提交 bug 修复、实验改进、文档优化和新模型实现。

## 本地开发

建议使用虚拟环境：

```bash
python -m venv .venv
pip install -r requirements.txt
```

运行完整流程：

```bash
python src/main.py
```

在无图形界面的环境中运行：

```bash
MPLBACKEND=Agg python src/main.py
```

运行测试：

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## 提交建议

- 优先复用现有模块和配置，不要在多个脚本中重复定义路径、阈值或超参数。
- 如果新增模型，请在 README 或文档中说明输入窗口、训练配置和评估指标。
- 如果修改预警逻辑，请补充或更新对应测试。
- 生成的数据和模型文件不要提交到仓库，它们已经由 `.gitignore` 忽略。

## Pull Request Checklist

- [ ] 代码能在本地运行
- [ ] 已运行测试
- [ ] README 或相关说明已同步更新
- [ ] 没有提交生成的模型、数据或缓存文件
