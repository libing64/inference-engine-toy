# 🔍 模型查看器和推理引擎

一个简易的PyTorch模型查看器和推理引擎，专注于可读性和易用性。

## 📋 项目目标

1. **模型查看器 (Model Viewer)**: 加载.pth格式的模型文件，并显示模型的详细结构
2. **推理引擎 (Inference Engine)**: 实现一个小型的模型推理引擎，不追求速度，优先保证可读性

## 🚀 功能特性

### 🔍 模型查看器
- ✅ 加载`.pth`格式的模型文件
- ✅ 显示模型概要信息（参数数量、层数等）
- ✅ 以树状结构展示模型架构
- ✅ 详细的层信息表格
- ✅ 支持state_dict格式的权重文件
- ✅ 导出模型信息到JSON文件

### ⚡ 推理引擎
- ✅ 逐层执行模型推理
- ✅ 实时显示推理过程和中间结果
- ✅ 支持常见的PyTorch层（Conv2d, Linear, ReLU等）
- ✅ 推理步骤详细记录和统计
- ✅ 模型性能基准测试
- ✅ 中间层输出获取功能

## 📦 环境要求

- Python 3.6+ （推荐Python 3.8+）
- PyTorch 1.12.0+

## 📦 安装依赖

```bash
# 使用Python 3安装依赖
conda create -n inference-engine-toy python=3.8
conda activate inference-engine-toy
pip3 install -r requirements.txt

# 或者使用用户模式安装
pip3 install torch torchvision numpy tabulate rich --user
```

## 🎯 快速开始

### 1. 运行演示程序
```bash
python3 demo.py
```

### 2. 交互式使用
```bash
python3 main.py --interactive
```

### 3. 快速分析模型
```bash
python3 main.py -m your_model.pth --quick
```

### 4. 运行单元测试
```bash
# 运行单元测试
python3 examples.py --test

# 测试模型查看器
python3 examples.py --test-viewer

# 测试推理引擎  
python3 examples.py --test-engine
```

## 📖 使用示例

### 基本用法

```python
from src.model_viewer import ModelViewer
from src.inference_engine import InferenceEngine
import torch

# 模型查看器
viewer = ModelViewer()
viewer.load_model('your_model.pth')
viewer.display_model_summary()
viewer.display_model_architecture()
viewer.display_layer_details()

# 推理引擎
engine = InferenceEngine(verbose=True)
engine.load_model(your_model)

input_data = torch.randn(1, 3, 224, 224)
result = engine.infer(input_data)

# 性能测试
stats = engine.benchmark_model(input_data, num_runs=10)
```

### 交互模式功能

启动交互模式后，你可以：

1. 🔍 **加载模型** - 从文件路径加载.pth模型
2. 📊 **查看模型信息** - 显示参数数量、层数等概要信息  
3. 🏗️ **查看模型架构** - 以树状图显示完整的模型结构
4. 📋 **查看层详情** - 详细的每层参数和配置信息
5. ⚡ **模型推理** - 输入测试数据，观察逐层推理过程
6. 🏃 **性能测试** - 多次运行测试，获取性能统计
7. 💾 **导出信息** - 将模型信息保存为JSON文件

## 📁 项目结构

```
inference-engine-toy/
├── src/                    # 源代码目录
│   ├── model_viewer.py     # 模型查看器实现
│   └── inference_engine.py # 推理引擎实现
├── main.py                 # 主程序入口
├── examples.py             # 示例和测试代码
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明
```

## 🧪 支持的模型层

推理引擎目前支持以下PyTorch层：

- **卷积层**: Conv2d, Conv1d
- **全连接层**: Linear
- **激活函数**: ReLU, Sigmoid, Tanh, Softmax
- **池化层**: MaxPool2d, AdaptiveAvgPool2d
- **归一化层**: BatchNorm2d, BatchNorm1d
- **其他**: Flatten, Dropout

## 🔧 命令行参数

### main.py
```bash
python main.py [选项]

选项:
  -m, --model MODEL     指定模型文件路径
  -i, --interactive     启动交互模式
  -q, --quick          快速分析模式
  --create-sample      创建示例模型
```

### examples.py  
```bash
python examples.py [选项]

选项:
  --demo              运行综合演示
  --test              运行单元测试
  --create-models     创建示例模型
  --test-viewer       测试模型查看器
  --test-engine       测试推理引擎
```

## 💡 设计理念

本项目优先考虑：
1. **可读性** - 代码结构清晰，逐步执行过程可视化
2. **易用性** - 简单的API设计，丰富的交互界面
3. **教育性** - 详细的推理过程展示，适合学习和调试

## 🔮 未来扩展

- [ ] 支持更多PyTorch层类型
- [ ] 添加模型可视化图形界面
- [ ] 支持ONNX格式模型
- [ ] 模型比较和分析功能
- [ ] Web界面支持

## 📄 许可证

MIT License