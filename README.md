# PyTorch 核心概念学习项目

这个项目旨在帮助你系统地学习 PyTorch 的核心概念，从基础知识到高级应用，采用渐进式学习方法。

## 项目结构

```
├── 01_tensor_basics/         # PyTorch 张量基础操作
├── 02_autograd/              # 自动微分机制
├── 03_neural_networks/       # 神经网络构建
├── 04_training/              # 模型训练与优化
├── 05_cnn/                   # 卷积神经网络
├── 06_rnn/                   # 循环神经网络
├── 07_transfer_learning/     # 迁移学习
├── 08_advanced_topics/       # 高级主题
└── data/                     # 数据集存放目录
```

## 学习路径

1. **张量基础**: 学习 PyTorch 的基本数据结构 - 张量，以及如何进行张量操作
2. **自动微分**: 理解 PyTorch 的自动微分机制，这是深度学习的核心
3. **神经网络**: 学习如何使用 PyTorch 构建神经网络模型
4. **训练过程**: 掌握模型训练、评估和保存的完整流程
5. **进阶模型**: 学习构建卷积神经网络(CNN)和循环神经网络(RNN)
6. **迁移学习**: 学习如何利用预训练模型
7. **高级主题**: 探索更多高级功能，如分布式训练、量化等

## 环境配置

```bash
# 创建虚拟环境
python -m venv pytorch_env

# 激活虚拟环境
# Windows
pytorch_env\Scripts\activate
# macOS/Linux
source pytorch_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

每个目录中的示例代码都是独立的，并且包含详细的注释。建议按照目录编号顺序学习，每个示例都可以直接运行：

```bash
python 01_tensor_basics/01_tensor_creation.py
```

## 学习资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 论坛](https://discuss.pytorch.org/)

## 贡献

如果你发现任何错误或有改进建议，欢迎提交 issue 或 pull request。

## 许可

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。