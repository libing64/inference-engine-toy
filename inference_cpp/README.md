# C++ 模型推理实现

基于 LibTorch (PyTorch C++ API) 实现静态模型的加载和推理，并与 PyTorch Python 推理进行对比。

## 📋 功能特性

1. **基于 C++ 实现静态模型的加载和推理**
   - 使用 LibTorch 加载 TorchScript 模型
   - 支持任意形状的输入数据
   - 自动性能统计和基准测试

2. **PyTorch vs C++ 推理对比**
   - 推理结果对比（形状、数值）
   - 推理速度对比（延迟、吞吐量）
   - 自动生成对比报告

## 🛠️ 环境要求

### 必需依赖

1. **LibTorch** (PyTorch C++ 库)
   - 下载地址: https://pytorch.org/get-started/locally/
   - 选择 Stable 版本，Platform: Linux，Language: C++
   - 下载后解压到本地目录（例如: `~/libtorch`）

2. **CMake** (>= 3.18)
   ```bash
   sudo apt-get install cmake
   ```

3. **C++ 编译器** (支持 C++17)
   ```bash
   sudo apt-get install g++ build-essential
   ```

4. **Python 依赖**
   ```bash
   pip install torch rich
   ```

## 📦 编译步骤

### 1. 设置 LibTorch 路径

**方法 A: 使用环境变量（推荐）**
```bash
export Torch_DIR=/path/to/libtorch/share/cmake/Torch
# 例如: export Torch_DIR=~/libtorch/share/cmake/Torch
```

**方法 B: 修改 CMakeLists.txt**
编辑 `CMakeLists.txt`，取消注释并设置路径：
```cmake
set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")
```

### 2. 编译 C++ 程序

```bash
cd inference_cpp
mkdir build
cd build
cmake ..
make
```

编译成功后，会在 `build/` 目录下生成 `inference` 可执行文件。

## 🚀 使用方法

### 1. 准备 TorchScript 模型

首先需要将 PyTorch 模型转换为 TorchScript 格式：

```bash
# 在项目根目录运行
python3 model_trace_demo.py
```

这会在 `traced_models/` 目录下生成 trace 后的模型文件，例如：
- `traced_models/simple_cnn_traced.pt`
- `traced_models/simple_resnet_traced.pt`

### 2. 运行 C++ 推理

```bash
cd build
./inference ../traced_models/simple_cnn_traced.pt 1,3,32,32
```

参数说明：
- 第一个参数：模型文件路径
- 第二个参数（可选）：输入形状，格式为 `batch,channel,height,width`，默认 `1,3,32,32`

### 3. 运行 PyTorch 推理（对比用）

```bash
cd inference_cpp
python3 pytorch_inference.py ../traced_models/simple_cnn_traced.pt 1,3,32,32
```

### 4. 运行对比脚本

```bash
cd inference_cpp
python3 compare_inference.py ../traced_models/simple_cnn_traced.pt build/inference
```

参数说明：
- 第一个参数：模型文件路径
- 第二个参数（可选）：C++ 可执行文件路径，默认 `./inference`

## 📊 输出示例

### C++ 推理输出

```
正在加载模型: ../traced_models/simple_cnn_traced.pt
✓ 模型加载成功!

=== 模型信息 ===
模型类型: TorchScript (ScriptModule)

=== 创建输入张量 ===
输入形状: [1, 3, 32, 32]

输入张量:
  形状: [1, 3, 32, 32]
  数据类型: float
  前 5 个元素: [0.123456, -0.234567, 0.345678, ...]

=== 执行推理 ===

输出张量:
  形状: [1, 10]
  数据类型: float
  前 5 个元素: [0.072443, -0.020700, 0.112696, ...]

=== 性能统计 ===
平均推理时间: 0.134 ms
吞吐量: 7462.69 FPS

✓ C++ 推理完成!
```

### 对比脚本输出

对比脚本会生成一个详细的对比表格，包括：
- 推理时间对比
- 吞吐量对比
- 输出形状一致性检查
- 输出数值差异分析

## 🔍 经典模型实例

项目包含以下经典模型的推理示例：

1. **SimpleCNN** - 简单的卷积神经网络
   - 模型文件: `traced_models/simple_cnn_traced.pt`
   - 输入形状: `(1, 3, 32, 32)`
   - 输出形状: `(1, 10)`

2. **SimpleResNet** - 简化版 ResNet
   - 模型文件: `traced_models/simple_resnet_traced.pt`
   - 输入形状: `(1, 3, 32, 32)`
   - 输出形状: `(1, 10)`

3. **ResNet18** - torchvision 的 ResNet18
   - 模型文件: `traced_models/resnet18_traced.pt`
   - 输入形状: `(1, 3, 224, 224)`
   - 输出形状: `(1, 1000)`

## 📈 性能对比结果

典型的对比结果（在相同硬件上）：

| 模型 | PyTorch (ms) | C++ (ms) | 加速比 |
|------|--------------|----------|--------|
| SimpleCNN | ~0.18 | ~0.13 | 1.4x |
| SimpleResNet | ~0.15 | ~0.12 | 1.25x |
| ResNet18 | ~5.2 | ~4.1 | 1.27x |

*注：实际性能取决于硬件配置、编译优化选项等因素*

## 🐛 常见问题

### 1. 找不到 LibTorch

**错误**: `Could not find a package configuration file provided by "Torch"`

**解决**: 确保设置了 `Torch_DIR` 环境变量，或修改 `CMakeLists.txt` 中的路径。

### 2. 编译错误: C++17 不支持

**错误**: `error: 'xxx' is not a member of 'std'`

**解决**: 确保编译器支持 C++17，检查 `CMakeLists.txt` 中的 `CMAKE_CXX_STANDARD` 设置。

### 3. 运行时错误: 找不到模型文件

**错误**: `Error: cannot load model`

**解决**: 检查模型文件路径是否正确，确保模型是 TorchScript 格式（通过 `torch.jit.trace` 生成）。

### 4. 输出数值不一致

**原因**: 浮点数精度差异、不同优化级别等。

**解决**: 这是正常现象，只要差异在可接受范围内（通常 < 1e-4）即可。

## 📚 参考资料

- [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)
- [TorchScript 教程](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [LibTorch 下载](https://pytorch.org/get-started/locally/)

## 📝 文件说明

- `inference.cpp` - C++ 推理主程序
- `CMakeLists.txt` - CMake 构建配置
- `pytorch_inference.py` - PyTorch 推理脚本（用于对比）
- `compare_inference.py` - 对比脚本
- `README.md` - 本文档