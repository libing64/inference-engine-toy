#!/bin/bash
# 构建脚本 - 自动编译 C++ 推理程序

set -e  # 遇到错误立即退出

echo "=========================================="
echo "C++ 推理程序构建脚本"
echo "=========================================="

# 检查 LibTorch 路径
if [ -z "$Torch_DIR" ]; then
    echo "警告: 未设置 Torch_DIR 环境变量"
    echo "请运行: export Torch_DIR=/path/to/libtorch/share/cmake/Torch"
    echo ""
    echo "或者，如果您已经下载了 LibTorch，请输入路径:"
    read -p "LibTorch 路径 (例如: ~/libtorch): " libtorch_path
    if [ ! -z "$libtorch_path" ]; then
        export Torch_DIR="$libtorch_path/share/cmake/Torch"
        echo "已设置 Torch_DIR=$Torch_DIR"
    else
        echo "错误: 必须设置 LibTorch 路径"
        exit 1
    fi
else
    echo "✓ 使用 Torch_DIR=$Torch_DIR"
fi

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake，请先安装: sudo apt-get install cmake"
    exit 1
fi

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++，请先安装: sudo apt-get install g++ build-essential"
    exit 1
fi

# 创建构建目录
echo ""
echo "创建构建目录..."
mkdir -p build
cd build

# 运行 CMake
echo ""
echo "运行 CMake..."
cmake ..

# 编译
echo ""
echo "开始编译..."
make

echo ""
echo "=========================================="
echo "✓ 编译完成！"
echo "=========================================="
echo ""
echo "可执行文件位置: $(pwd)/inference"
echo ""
echo "使用方法:"
echo "  ./inference <模型路径> [输入形状]"
echo "  示例: ./inference ../traced_models/simple_cnn_traced.pt 1,3,32,32"
echo ""
