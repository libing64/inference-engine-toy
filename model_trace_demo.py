# -*- coding: utf-8 -*-
"""
模型 Trace 演示脚本 - 展示如何将动态图模型转换为静态图
Model Trace Demo - Demonstrates how to convert dynamic graph models to static graph
"""

import torch
import torch.nn as nn
import torchvision.models as models
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os

console = Console()

# 导入示例模型类
try:
    from examples import SimpleCNN, SimpleMLP, SimpleResNet
except ImportError:
    # 如果导入失败，定义简单的模型类
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


def trace_model(model, example_input, model_name, output_dir="traced_models"):
    """
    对模型进行 TorchScript trace 并保存
    
    Args:
        model: PyTorch 模型
        example_input: 示例输入数据
        model_name: 模型名称（用于保存文件）
        output_dir: 输出目录
        
    Returns:
        traced_model: trace 后的模型
    """
    console.print(f"\n[bold yellow]正在 Trace 模型: {model_name}[/bold yellow]")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置为评估模式
    model.eval()
    
    try:
        # 执行 trace
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # 保存 trace 后的模型
        output_path = os.path.join(output_dir, f"{model_name}_traced.pt")
        traced_model.save(output_path)
        
        console.print(f"[green]✓ Trace 成功！已保存到: {output_path}[/green]")
        
        # 验证 trace 后的模型
        console.print(f"[cyan]验证 trace 后的模型...[/cyan]")
        loaded_model = torch.jit.load(output_path)
        
        # 测试推理
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = loaded_model(example_input)
        
        # 检查输出是否一致
        if torch.allclose(original_output, traced_output, atol=1e-5):
            console.print(f"[green]✓ 验证通过：trace 后的模型输出与原始模型一致[/green]")
        else:
            console.print(f"[yellow]⚠ 警告：trace 后的模型输出与原始模型有微小差异[/yellow]")
        
        return traced_model, output_path
        
    except Exception as e:
        console.print(f"[red]✗ Trace 失败: {str(e)}[/red]")
        return None, None


def demo_simple_cnn():
    """演示：Trace 简单的 CNN 模型"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]演示 1: 简单 CNN 模型[/bold cyan]")
    console.print("=" * 60)
    
    model = SimpleCNN(num_classes=10)
    example_input = torch.randn(1, 3, 32, 32)
    
    traced_model, path = trace_model(model, example_input, "simple_cnn")
    
    if traced_model:
        # 显示模型信息
        console.print(f"\n[bold]模型信息:[/bold]")
        console.print(f"  原始模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        console.print(f"  Trace 后模型大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


def demo_resnet():
    """演示：Trace ResNet 模型"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]演示 2: ResNet 模型[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # 使用 torchvision 的预训练 ResNet
        model = models.resnet18(pretrained=False)  # 使用 False 避免下载
        model.eval()
        
        example_input = torch.randn(1, 3, 224, 224)
        
        traced_model, path = trace_model(model, example_input, "resnet18")
        
        if traced_model:
            console.print(f"\n[bold]模型信息:[/bold]")
            console.print(f"  原始模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            console.print(f"  Trace 后模型大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
    except Exception as e:
        console.print(f"[red]无法加载 ResNet: {str(e)}[/red]")


def demo_custom_resnet():
    """演示：Trace 自定义的 SimpleResNet"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]演示 3: 自定义 SimpleResNet[/bold cyan]")
    console.print("=" * 60)
    
    try:
        model = SimpleResNet(num_classes=10)
        example_input = torch.randn(1, 3, 32, 32)
        
        traced_model, path = trace_model(model, example_input, "simple_resnet")
        
        if traced_model:
            console.print(f"\n[bold]模型信息:[/bold]")
            console.print(f"  原始模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            console.print(f"  Trace 后模型大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
    except NameError:
        console.print("[yellow]SimpleResNet 未定义，跳过此演示[/yellow]")


def demo_mlp():
    """演示：Trace MLP 模型"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]演示 4: MLP 模型[/bold cyan]")
    console.print("=" * 60)
    
    try:
        model = SimpleMLP(input_size=784, num_classes=10)
        example_input = torch.randn(1, 784)  # MNIST 尺寸
        
        traced_model, path = trace_model(model, example_input, "simple_mlp")
        
        if traced_model:
            console.print(f"\n[bold]模型信息:[/bold]")
            console.print(f"  原始模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            console.print(f"  Trace 后模型大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
    except NameError:
        console.print("[yellow]SimpleMLP 未定义，跳过此演示[/yellow]")


def compare_models():
    """对比原始模型和 trace 后的模型"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]对比：原始模型 vs Trace 后的模型[/bold cyan]")
    console.print("=" * 60)
    
    model = SimpleCNN(num_classes=10)
    example_input = torch.randn(1, 3, 32, 32)
    
    # Trace 模型
    traced_model, path = trace_model(model, example_input, "comparison_test")
    
    if not traced_model:
        return
    
    # 创建对比表格
    table = Table(title="模型对比")
    table.add_column("特性", style="cyan")
    table.add_column("原始模型", style="magenta")
    table.add_column("Trace 后模型", style="green")
    
    # 推理速度对比（简单测试）
    import time
    
    model.eval()
    traced_model.eval()
    
    # 预热
    for _ in range(5):
        _ = model(example_input)
        _ = traced_model(example_input)
    
    # 测试原始模型
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(example_input)
    original_time = (time.time() - start) / 100 * 1000  # ms
    
    # 测试 trace 后的模型
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = traced_model(example_input)
    traced_time = (time.time() - start) / 100 * 1000  # ms
    
    table.add_row("类型", "nn.Module", "torch.jit.ScriptModule")
    table.add_row("可序列化", "需要代码", "✓ 独立文件")
    table.add_row("推理速度 (ms)", f"{original_time:.3f}", f"{traced_time:.3f}")
    table.add_row("文件大小", "N/A (代码)", f"{os.path.getsize(path) / 1024:.2f} KB")
    table.add_row("跨平台", "需要 Python", "✓ C++/移动端")
    
    console.print(table)


def load_and_test_traced_model():
    """演示：加载并使用 trace 后的模型"""
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]演示：加载 Trace 后的模型[/bold cyan]")
    console.print("=" * 60)
    
    traced_model_path = "traced_models/simple_cnn_traced.pt"
    
    if not os.path.exists(traced_model_path):
        console.print(f"[yellow]未找到 trace 后的模型文件: {traced_model_path}[/yellow]")
        console.print("[yellow]请先运行前面的演示生成模型文件[/yellow]")
        return
    
    try:
        # 加载 trace 后的模型（不需要原始模型定义！）
        console.print(f"[cyan]正在加载: {traced_model_path}[/cyan]")
        loaded_model = torch.jit.load(traced_model_path)
        
        # 测试推理
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = loaded_model(test_input)
        
        console.print(f"[green]✓ 成功加载并运行！[/green]")
        console.print(f"  输入形状: {list(test_input.shape)}")
        console.print(f"  输出形状: {list(output.shape)}")
        console.print(f"  输出示例: {output[0, :5].tolist()}")
        
        # 显示模型结构（trace 后的模型也有 graph）
        console.print(f"\n[bold]模型结构（部分）:[/bold]")
        console.print(f"  {str(loaded_model.graph)[:200]}...")
        
    except Exception as e:
        console.print(f"[red]加载失败: {str(e)}[/red]")


def main():
    """主函数"""
    welcome_text = """
    [bold green]TorchScript Trace 演示[/bold green]
    
    本脚本将演示如何将 PyTorch 动态图模型转换为静态图（TorchScript）。
    
    [bold]主要功能:[/bold]
    • 使用 torch.jit.trace 转换模型
    • 保存 trace 后的模型文件
    • 验证 trace 后的模型正确性
    • 对比原始模型和 trace 后模型的特性
    """
    
    console.print(Panel(welcome_text, title="欢迎", border_style="green"))
    
    # 运行各种演示
    demo_simple_cnn()
    demo_mlp()
    demo_custom_resnet()
    demo_resnet()
    compare_models()
    load_and_test_traced_model()
    
    console.print("\n[bold green]🎉 所有演示完成！[/bold green]")
    console.print(f"\n[cyan]Trace 后的模型文件保存在: traced_models/ 目录[/cyan]")
    console.print("[cyan]💡 提示:[/cyan]")
    console.print("  • Trace 后的模型可以在没有原始代码的情况下加载")
    console.print("  • 适合部署到 C++ 环境或移动端")
    console.print("  • 注意：trace 只记录一次执行路径，不适合动态控制流")


if __name__ == "__main__":
    main()
