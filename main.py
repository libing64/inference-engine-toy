# -*- coding: utf-8 -*-
"""
主程序入口 - 模型查看器和推理引擎的统一界面
Main Entry Point - Unified interface for model viewer and inference engine
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from typing import Optional

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_viewer import ModelViewer
from inference_engine import InferenceEngine

# 导入示例模型类
try:
    from examples import SimpleCNN, SimpleMLP, SimpleResNet
except ImportError:
    # 如果导入失败，定义一个简单的模型类
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)


class ModelAnalyzer:
    """模型分析器 - 整合查看器和推理引擎"""
    
    def __init__(self):
        self.console = Console()
        self.viewer = ModelViewer()
        self.engine = InferenceEngine(verbose=True)
        self.current_model = None
        
    def welcome_message(self):
        """显示欢迎信息"""
        welcome_text = Text()
        welcome_text.append("🔍 ", style="bold blue")
        welcome_text.append("模型查看器和推理引擎", style="bold green")
        welcome_text.append(" 🚀\n\n", style="bold blue")
        welcome_text.append("功能说明:\n", style="bold")
        welcome_text.append("• 加载并查看PyTorch模型结构\n", style="cyan")
        welcome_text.append("• 执行模型推理并显示详细过程\n", style="cyan")
        welcome_text.append("• 性能基准测试\n", style="cyan")
        welcome_text.append("• 导出模型信息\n", style="cyan")
        
        panel = Panel(welcome_text, title="欢迎使用", border_style="green")
        self.console.print(panel)
    
    def interactive_mode(self):
        """交互式模式"""
        self.welcome_message()
        
        while True:
            self.console.print("\n" + "="*50)
            self.console.print("[bold yellow]请选择操作:[/bold yellow]")
            self.console.print("1. 🔍 加载模型")
            self.console.print("2. 📊 查看模型信息")
            self.console.print("3. 🏗️  查看模型架构")
            self.console.print("4. 📋 查看层详情")
            self.console.print("5. 📏 追踪模型形状")
            self.console.print("6. ⚡ 模型推理")
            self.console.print("7. 🏃 性能测试")
            self.console.print("8. 💾 导出信息")
            self.console.print("0. 👋 退出程序")
            
            choice = Prompt.ask("请输入选择", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            
            try:
                if choice == "0":
                    self.console.print("[green]👋 再见![/green]")
                    break
                elif choice == "1":
                    self._handle_load_model()
                elif choice == "2":
                    self._handle_model_info()
                elif choice == "3":
                    self._handle_model_architecture()
                elif choice == "4":
                    self._handle_layer_details()
                elif choice == "5":
                    self._handle_trace_shapes()
                elif choice == "6":
                    self._handle_inference()
                elif choice == "7":
                    self._handle_benchmark()
                elif choice == "8":
                    self._handle_export()
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]操作被用户中断[/yellow]")
            except Exception as e:
                self.console.print(f"[red]操作出错: {str(e)}[/red]")
    
    def _handle_load_model(self):
        """处理模型加载"""
        model_path = Prompt.ask("请输入模型文件路径 (.pth)")
        
        if not os.path.exists(model_path):
            self.console.print("[red]文件不存在![/red]")
            return
        
        # 加载到查看器
        if self.viewer.load_model(model_path):
            self.console.print("[green]✓ 模型已加载到查看器[/green]")
            
            # 尝试加载到推理引擎
            if hasattr(self.viewer, 'model') and self.viewer.model is not None:
                self.engine.load_model(self.viewer.model)
                self.current_model = self.viewer.model
                self.console.print("[green]✓ 模型已加载到推理引擎[/green]")
            else:
                self.console.print("[yellow]⚠️ 只有权重文件，推理功能需要完整模型[/yellow]")
        else:
            self.console.print("[red]模型加载失败![/red]")
    
    def _handle_model_info(self):
        """处理模型信息显示"""
        if not self._check_model_loaded():
            return
        
        self.viewer.display_model_summary()
        
        # 如果只有state_dict，也显示相关信息
        if 'state_dict' in self.viewer.model_info:
            self.viewer.display_state_dict_info()
    
    def _handle_model_architecture(self):
        """处理模型架构显示"""
        if not self._check_model_loaded():
            return
        
        if hasattr(self.viewer, 'model') and self.viewer.model is not None:
            self.viewer.display_model_architecture()
        else:
            self.console.print("[yellow]只有权重信息，无法显示完整架构[/yellow]")
            self.viewer.display_state_dict_info()
    
    def _ensure_shape_info(self):
        """确保模型包含形状信息"""
        # 检查是否已经有形状信息（通过检查第一层的input_shape是否为Unknown）
        if self.viewer.model_info.get('layers') and \
           self.viewer.model_info['layers'][0].get('input_shape') == 'Unknown':
            
            if Confirm.ask("模型缺少输入输出形状信息，是否现在进行追踪?", default=True):
                try:
                    input_shape = Prompt.ask("请输入输入数据形状 (例如: 1,3,224,224)")
                    shape_list = [int(x.strip()) for x in input_shape.split(',')]
                    self.viewer.trace_model_shapes(tuple(shape_list))
                except ValueError:
                    self.console.print("[red]输入格式无效，跳过形状追踪[/red]")

    def _handle_layer_details(self):
        """处理层详情显示"""
        if not self._check_model_loaded():
            return
            
        self._ensure_shape_info()
        self.viewer.display_layer_details()

    def _handle_trace_shapes(self):
        """处理模型形状追踪"""
        if not self._check_model_loaded():
            return
            
        try:
            input_shape = Prompt.ask("请输入输入数据形状 (例如: 1,3,224,224)")
            shape_list = [int(x.strip()) for x in input_shape.split(',')]
            self.viewer.trace_model_shapes(tuple(shape_list))
        except ValueError:
            self.console.print("[red]输入格式无效[/red]")

    def _handle_inference(self):
        """处理模型推理"""
        if not self._check_inference_ready():
            return
        
        # 获取输入参数
        try:
            input_shape = Prompt.ask("请输入数据形状 (例如: 1,3,224,224)")
            shape_list = [int(x.strip()) for x in input_shape.split(',')]
            
            # 创建随机输入数据
            input_data = torch.randn(*shape_list)
            self.console.print(f"[green]已创建随机输入数据，形状: {list(input_data.shape)}[/green]")
            
            # 执行推理
            result = self.engine.infer(input_data, detailed=True)
            self.console.print(f"\n[bold green]推理结果形状: {list(result.shape)}[/bold green]")
            
            # 询问是否显示结果数据
            if Confirm.ask("是否显示输出数据?"):
                self.console.print(f"输出数据:\n{result}")
            
        except ValueError as e:
            self.console.print(f"[red]输入格式错误: {str(e)}[/red]")
        except Exception as e:
            self.console.print(f"[red]推理失败: {str(e)}[/red]")
    
    def _handle_benchmark(self):
        """处理性能测试"""
        if not self._check_inference_ready():
            return
        
        try:
            input_shape = Prompt.ask("请输入测试数据形状 (例如: 1,3,224,224)")
            shape_list = [int(x.strip()) for x in input_shape.split(',')]
            
            num_runs = int(Prompt.ask("请输入测试次数", default="10"))
            
            # 创建测试数据
            input_data = torch.randn(*shape_list)
            
            # 执行基准测试
            stats = self.engine.benchmark_model(input_data, num_runs)
            
        except ValueError as e:
            self.console.print(f"[red]输入错误: {str(e)}[/red]")
        except Exception as e:
            self.console.print(f"[red]基准测试失败: {str(e)}[/red]")
    
    def _handle_export(self):
        """处理信息导出"""
        if not self._check_model_loaded():
            return
        
        self._ensure_shape_info()
        output_path = Prompt.ask("请输入导出文件路径", default="model_info.json")
        self.viewer.export_model_info(output_path)
    
    def _check_model_loaded(self) -> bool:
        """检查是否已加载模型"""
        if not self.viewer.model_info:
            self.console.print("[red]请先加载模型![/red]")
            return False
        return True
    
    def _check_inference_ready(self) -> bool:
        """检查是否可以执行推理"""
        if not hasattr(self.engine, 'model'):
            self.console.print("[red]请先加载完整的模型文件![/red]")
            return False
        return True
    
    def load_model_from_path(self, model_path: str) -> bool:
        """从命令行直接加载模型"""
        return self.viewer.load_model(model_path)
    
    def quick_analysis(self, model_path: str):
        """快速分析模式"""
        if self.load_model_from_path(model_path):
            self.console.print("\n[bold yellow]📊 模型信息概览:[/bold yellow]")
            self.viewer.display_model_summary()
            
            self.console.print("\n[bold yellow]🏗️ 模型架构:[/bold yellow]")
            if hasattr(self.viewer, 'model') and self.viewer.model is not None:
                self.viewer.display_model_architecture()
            else:
                self.viewer.display_state_dict_info()


def create_sample_model():
    """创建一个示例模型用于测试"""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 10),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return SimpleNet()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型查看器和推理引擎")
    parser.add_argument('--model', '-m', type=str, help='模型文件路径')
    parser.add_argument('--interactive', '-i', action='store_true', help='启动交互模式')
    parser.add_argument('--quick', '-q', action='store_true', help='快速分析模式')
    parser.add_argument('--create-sample', action='store_true', help='创建示例模型')
    
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer()
    
    # 创建示例模型
    if args.create_sample:
        sample_model = create_sample_model()
        sample_path = "sample_model.pth"
        torch.save(sample_model, sample_path)
        analyzer.console.print(f"[green]✓ 示例模型已保存到: {sample_path}[/green]")
        return
    
    # 如果指定了模型文件
    if args.model:
        if args.quick:
            # 快速分析模式
            analyzer.quick_analysis(args.model)
        else:
            # 加载模型后进入交互模式
            if analyzer.load_model_from_path(args.model):
                if args.interactive:
                    analyzer.interactive_mode()
            else:
                analyzer.console.print("[red]模型加载失败![/red]")
                sys.exit(1)
    else:
        # 默认进入交互模式
        analyzer.interactive_mode()


if __name__ == "__main__":
    main()
