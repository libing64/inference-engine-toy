# -*- coding: utf-8 -*-
"""
快速演示脚本 - 展示模型查看器和推理引擎的核心功能
Quick Demo - Demonstrates core functionality of model viewer and inference engine
"""

import torch
import torch.nn as nn
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_viewer import ModelViewer
from inference_engine import InferenceEngine
from rich.console import Console

console = Console()


class SimpleCNN(nn.Module):
    """简单的CNN模型用于演示"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    """主演示函数"""
    
    console.print("[bold green]🚀 模型查看器和推理引擎演示[/bold green]")
    console.print("="*60)
    
    # 1. 创建示例模型
    console.print("\n[bold yellow]1️⃣ 创建示例模型[/bold yellow]")
    model = SimpleCNN(num_classes=10)
    console.print("[green]✓ SimpleCNN模型创建成功[/green]")
    
    # 2. 测试模型查看器
    console.print("\n[bold yellow]2️⃣ 模型查看器功能演示[/bold yellow]")
    viewer = ModelViewer()
    
    # 直接设置模型（跳过文件加载）
    viewer.model = model
    viewer._collect_model_info()
    
    console.print("\n[cyan]--- 模型概要信息 ---[/cyan]")
    viewer.display_model_summary()
    
    console.print("\n[cyan]--- 模型架构 ---[/cyan]")
    viewer.display_model_architecture()
    
    console.print("\n[cyan]--- 层详情信息 ---[/cyan]")
    viewer.display_layer_details()
    
    # 3. 测试推理引擎
    console.print("\n[bold yellow]3️⃣ 推理引擎功能演示[/bold yellow]")
    engine = InferenceEngine(verbose=True)
    engine.load_model(model)
    
    # 创建测试输入
    console.print("\n[cyan]--- 执行推理 ---[/cyan]")
    input_data = torch.randn(1, 3, 32, 32)  # CIFAR-10 尺寸
    console.print(f"输入数据形状: {list(input_data.shape)}")
    
    result = engine.infer(input_data, detailed=True)
    console.print(f"\n[bold green]推理结果形状: {list(result.shape)}[/bold green]")
    console.print(f"预测结果: {result.argmax(dim=1).item()}")
    
    # 4. 性能测试
    console.print("\n[bold yellow]4️⃣ 性能基准测试[/bold yellow]")
    stats = engine.benchmark_model(input_data, num_runs=5)
    
    # 5. 中间层输出测试
    console.print("\n[bold yellow]5️⃣ 中间层输出获取[/bold yellow]")
    steps = engine.get_inference_steps()
    console.print(f"推理步骤数量: {len(steps)}")
    
    if steps:
        first_step = steps[0]
        console.print(f"第一层: {first_step.layer_name} ({first_step.operation})")
        console.print(f"输入形状: {first_step.input_shape} → 输出形状: {first_step.output_shape}")
    
    console.print("\n[bold green]🎉 演示完成![/bold green]")
    console.print("\n[cyan]💡 使用提示:[/cyan]")
    console.print("• 本项目支持加载.pth/.pt格式的模型文件")
    console.print("• 运行 'python3 main.py --interactive' 进入交互模式")
    console.print("• 推理引擎优先考虑可读性，显示详细的执行过程")


if __name__ == "__main__":
    main()
