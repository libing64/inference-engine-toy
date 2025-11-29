# -*- coding: utf-8 -*-
"""
示例和测试代码 - 展示模型查看器和推理引擎的使用方法
Examples and Test Code - Demonstrates the usage of model viewer and inference engine
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_viewer import ModelViewer
from inference_engine import InferenceEngine
from rich.console import Console

console = Console()


def create_simple_cnn():
    """创建一个简单的CNN模型"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            
            # 特征提取层
            self.features = nn.Sequential(
                # 第一个卷积块
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # 第二个卷积块
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # 第三个卷积块
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return SimpleCNN()


def create_simple_mlp():
    """创建一个简单的多层感知机"""
    class SimpleMLP(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
            super(SimpleMLP, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2)
                ])
                prev_size = hidden_size
            
            # 输出层
            layers.append(nn.Linear(prev_size, num_classes))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    return SimpleMLP()


def create_resnet_like():
    """创建一个类似ResNet的简化模型"""
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                 kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # 跳跃连接
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = nn.ReLU()(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = nn.ReLU()(out)
            return out
    
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleResNet, self).__init__()
            
            # 初始卷积层
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            
            # ResNet块
            self.layer1 = BasicBlock(16, 16)
            self.layer2 = BasicBlock(16, 32, stride=2)
            self.layer3 = BasicBlock(32, 64, stride=2)
            
            # 全局平均池化和分类器
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = nn.ReLU()(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    return SimpleResNet()


def save_example_models():
    """保存示例模型到文件"""
    console.print("[yellow]正在创建示例模型...[/yellow]")
    
    # 创建models目录
    models_dir = "example_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存不同类型的模型
    models_to_save = {
        "simple_cnn.pth": create_simple_cnn(),
        "simple_mlp.pth": create_simple_mlp(),
        "simple_resnet.pth": create_resnet_like(),
    }
    
    for filename, model in models_to_save.items():
        file_path = os.path.join(models_dir, filename)
        
        # 保存完整模型
        torch.save(model, file_path)
        console.print(f"[green]✓ 已保存: {file_path}[/green]")
        
        # 也保存state_dict版本
        state_dict_path = file_path.replace('.pth', '_state_dict.pth')
        torch.save(model.state_dict(), state_dict_path)
        console.print(f"[green]✓ 已保存: {state_dict_path}[/green]")
    
    # 尝试保存一个预训练模型的state_dict（如果可用）
    try:
        resnet18 = models.resnet18(pretrained=False)  # 使用False避免下载
        torch.save(resnet18, os.path.join(models_dir, "resnet18_example.pth"))
        console.print(f"[green]✓ 已保存: {os.path.join(models_dir, 'resnet18_example.pth')}[/green]")
    except Exception as e:
        console.print(f"[yellow]注意: 无法保存ResNet18示例 ({str(e)})[/yellow]")
    
    console.print(f"\n[bold green]✅ 示例模型已保存到 {models_dir}/ 目录[/bold green]")


def test_model_viewer():
    """测试模型查看器功能"""
    console.print("\n[bold blue]🔍 测试模型查看器功能[/bold blue]")
    
    # 创建测试模型
    model = create_simple_cnn()
    model_path = "test_model.pth"
    torch.save(model, model_path)
    
    # 创建查看器并测试
    viewer = ModelViewer()
    
    console.print(f"\n[yellow]测试加载模型: {model_path}[/yellow]")
    if viewer.load_model(model_path):
        console.print("\n[cyan]--- 模型概要信息 ---[/cyan]")
        viewer.display_model_summary()
        
        console.print("\n[cyan]--- 模型架构 ---[/cyan]")
        viewer.display_model_architecture()
        
        console.print("\n[cyan]--- 层详情 ---[/cyan]")
        viewer.display_layer_details()
        
        console.print("\n[cyan]--- 导出信息测试 ---[/cyan]")
        viewer.export_model_info("test_model_info.json")
    
    # 清理测试文件
    if os.path.exists(model_path):
        os.remove(model_path)


def test_inference_engine():
    """测试推理引擎功能"""
    console.print("\n[bold blue]⚡ 测试推理引擎功能[/bold blue]")
    
    # 创建测试模型
    model = create_simple_cnn()
    
    # 创建推理引擎并测试
    engine = InferenceEngine(verbose=True)
    engine.load_model(model)
    
    # 测试推理
    console.print("\n[yellow]测试推理功能...[/yellow]")
    input_data = torch.randn(1, 3, 32, 32)  # CIFAR-10尺寸
    
    result = engine.infer(input_data, detailed=True)
    console.print(f"\n[green]推理结果形状: {list(result.shape)}[/green]")
    
    # 测试基准测试
    console.print("\n[yellow]测试基准测试功能...[/yellow]")
    stats = engine.benchmark_model(input_data, num_runs=5)


def test_mlp_model():
    """测试MLP模型"""
    console.print("\n[bold blue]🧠 测试MLP模型[/bold blue]")
    
    model = create_simple_mlp()
    
    # 测试查看器
    viewer = ModelViewer()
    viewer.model = model
    viewer._collect_model_info()
    viewer.display_model_summary()
    viewer.display_model_architecture()
    
    # 测试推理引擎
    engine = InferenceEngine(verbose=True)
    engine.load_model(model)
    
    # 创建适合MLP的输入（展平的图像数据）
    input_data = torch.randn(1, 784)  # MNIST尺寸
    result = engine.infer(input_data)
    console.print(f"MLP推理结果形状: {list(result.shape)}")


def comprehensive_demo():
    """综合演示"""
    console.print("\n" + "="*60)
    console.print("[bold green]🚀 模型查看器和推理引擎综合演示[/bold green]")
    console.print("="*60)
    
    # 1. 创建并保存示例模型
    console.print("\n[bold yellow]1️⃣ 创建示例模型[/bold yellow]")
    save_example_models()
    
    # 2. 测试模型查看器
    console.print("\n[bold yellow]2️⃣ 测试模型查看器[/bold yellow]")
    test_model_viewer()
    
    # 3. 测试推理引擎
    console.print("\n[bold yellow]3️⃣ 测试推理引擎[/bold yellow]")
    test_inference_engine()
    
    # 4. 测试MLP模型
    console.print("\n[bold yellow]4️⃣ 测试MLP模型[/bold yellow]")
    test_mlp_model()
    
    console.print("\n[bold green]✅ 综合演示完成![/bold green]")
    console.print("\n[cyan]💡 使用提示:[/cyan]")
    console.print("• 运行 'python main.py --interactive' 进入交互模式")
    console.print("• 运行 'python main.py --create-sample' 创建示例模型")
    console.print("• 运行 'python main.py -m model.pth --quick' 快速分析模型")


def run_unit_tests():
    """运行单元测试"""
    console.print("\n[bold blue]🧪 运行单元测试[/bold blue]")
    
    tests_passed = 0
    tests_total = 0
    
    # 测试1: 模型创建
    tests_total += 1
    try:
        model = create_simple_cnn()
        assert isinstance(model, nn.Module)
        console.print("[green]✓ 测试1: 模型创建成功[/green]")
        tests_passed += 1
    except Exception as e:
        console.print(f"[red]✗ 测试1: 模型创建失败 - {str(e)}[/red]")
    
    # 测试2: 模型查看器初始化
    tests_total += 1
    try:
        viewer = ModelViewer()
        assert viewer is not None
        console.print("[green]✓ 测试2: 模型查看器初始化成功[/green]")
        tests_passed += 1
    except Exception as e:
        console.print(f"[red]✗ 测试2: 模型查看器初始化失败 - {str(e)}[/red]")
    
    # 测试3: 推理引擎初始化
    tests_total += 1
    try:
        engine = InferenceEngine()
        assert engine is not None
        console.print("[green]✓ 测试3: 推理引擎初始化成功[/green]")
        tests_passed += 1
    except Exception as e:
        console.print(f"[red]✗ 测试3: 推理引擎初始化失败 - {str(e)}[/red]")
    
    # 测试4: 模型前向传播
    tests_total += 1
    try:
        model = create_simple_cnn()
        input_data = torch.randn(1, 3, 32, 32)
        output = model(input_data)
        assert output.shape == (1, 10)
        console.print("[green]✓ 测试4: 模型前向传播成功[/green]")
        tests_passed += 1
    except Exception as e:
        console.print(f"[red]✗ 测试4: 模型前向传播失败 - {str(e)}[/red]")
    
    console.print(f"\n[bold]测试结果: {tests_passed}/{tests_total} 通过[/bold]")
    if tests_passed == tests_total:
        console.print("[bold green]🎉 所有测试通过![/bold green]")
    else:
        console.print("[bold red]❌ 部分测试失败[/bold red]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="示例和测试程序")
    parser.add_argument('--demo', action='store_true', help='运行综合演示')
    parser.add_argument('--test', action='store_true', help='运行单元测试')
    parser.add_argument('--create-models', action='store_true', help='创建示例模型')
    parser.add_argument('--test-viewer', action='store_true', help='测试模型查看器')
    parser.add_argument('--test-engine', action='store_true', help='测试推理引擎')
    
    args = parser.parse_args()
    
    if args.demo:
        comprehensive_demo()
    elif args.test:
        run_unit_tests()
    elif args.create_models:
        save_example_models()
    elif args.test_viewer:
        test_model_viewer()
    elif args.test_engine:
        test_inference_engine()
    else:
        # 默认运行综合演示
        comprehensive_demo()
