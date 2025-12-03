# -*- coding: utf-8 -*-
"""
PyTorch 推理脚本 - 用于与 C++ 推理进行对比
PyTorch Inference Script - For comparison with C++ inference
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加父目录到路径以导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from examples import SimpleCNN
except ImportError:
    # 如果导入失败，定义简单的模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Linear(32 * 16 * 16, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


def pytorch_inference(model_path, input_shape=(1, 3, 32, 32), warmup=5, runs=100):
    """
    使用 PyTorch 进行推理
    
    Args:
        model_path: 模型文件路径（.pt 或 .pth）
        input_shape: 输入数据形状
        warmup: 预热次数
        runs: 正式测试次数
        
    Returns:
        (output_tensor, avg_time_ms): 输出张量和平均推理时间（毫秒）
    """
    print(f"正在加载模型: {model_path}")
    
    # 加载模型
    if model_path.endswith('_traced.pt'):
        # TorchScript 模型
        model = torch.jit.load(model_path)
    else:
        # 普通 PyTorch 模型
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            raise ValueError("无法从文件中提取模型")
    
    model.eval()
    print("✓ 模型加载成功!")
    
    # 创建输入数据
    input_data = torch.randn(*input_shape)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
    
    # 正式计时
    start_time = time.time()
    output = None
    with torch.no_grad():
        for _ in range(runs):
            output = model(input_data)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / runs * 1000
    
    return output, avg_time_ms


def print_tensor_info(tensor, name):
    """打印张量信息"""
    print(f"\n{name}:")
    print(f"  形状: {list(tensor.shape)}")
    print(f"  数据类型: {tensor.dtype}")
    flat = tensor.flatten()
    num_elements = min(5, flat.numel())
    print(f"  前 {num_elements} 个元素: {flat[:num_elements].tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python pytorch_inference.py <模型路径> [输入形状]")
        print("示例: python pytorch_inference.py ../traced_models/simple_cnn_traced.pt 1,3,32,32")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # 解析输入形状
    if len(sys.argv) >= 3:
        input_shape = tuple(int(x) for x in sys.argv[2].split(','))
    else:
        input_shape = (1, 3, 32, 32)
    
    print("=== PyTorch 推理 ===")
    print(f"输入形状: {input_shape}")
    
    try:
        output, avg_time = pytorch_inference(model_path, input_shape)
        
        print_tensor_info(output, "输出张量")
        
        print("\n=== 性能统计 ===")
        print(f"平均推理时间: {avg_time:.3f} ms")
        print(f"吞吐量: {1000.0 / avg_time:.2f} FPS")
        
        print("\n✓ PyTorch 推理完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
