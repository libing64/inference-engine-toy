#!/bin/bash
# 完整演示脚本 - 展示从模型trace到推理对比的完整流程

set -e

echo "=========================================="
echo "C++ 推理完整演示"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "步骤 1: 生成 TorchScript 模型（如果不存在）"
echo "----------------------------------------"
if [ ! -d "$PROJECT_ROOT/traced_models" ] || [ -z "$(ls -A $PROJECT_ROOT/traced_models/*.pt 2>/dev/null)" ]; then
    echo "未找到 trace 后的模型，正在生成..."
    cd "$PROJECT_ROOT"
    python3 model_trace_demo.py
else
    echo "✓ 已存在 trace 后的模型文件"
fi

echo ""
echo "步骤 2: 编译 C++ 推理程序（如果未编译）"
echo "----------------------------------------"
cd "$SCRIPT_DIR"
if [ ! -f "build/inference" ]; then
    echo "未找到编译后的程序，正在编译..."
    if [ -z "$Torch_DIR" ]; then
        echo "警告: 未设置 Torch_DIR，请先设置 LibTorch 路径"
        echo "运行: export Torch_DIR=/path/to/libtorch/share/cmake/Torch"
        echo "或者运行: ./build.sh"
        exit 1
    fi
    ./build.sh
else
    echo "✓ 已存在编译后的程序"
fi

echo ""
echo "步骤 3: 运行 PyTorch 推理"
echo "----------------------------------------"
MODEL_PATH="$PROJECT_ROOT/traced_models/simple_cnn_traced.pt"
python3 pytorch_inference.py "$MODEL_PATH" 1,3,32,32

echo ""
echo "步骤 4: 运行 C++ 推理"
echo "----------------------------------------"
cd build
./inference "$MODEL_PATH" 1,3,32,32

echo ""
echo "步骤 5: 运行对比测试"
echo "----------------------------------------"
cd ..
python3 compare_inference.py "$MODEL_PATH" build/inference

echo ""
echo "=========================================="
echo "✓ 演示完成！"
echo "=========================================="
