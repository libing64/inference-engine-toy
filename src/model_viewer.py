# -*- coding: utf-8 -*-
"""
模型查看器 - 用于加载和显示PyTorch模型的结构信息
Model Viewer - For loading and displaying PyTorch model structure information
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
import os


class ModelViewer:
    """简易模型查看器，用于加载.pth格式的模型文件并显示模型结构"""
    
    def __init__(self):
        self.console = Console()
        self.model = None
        self.model_info = {}
    
    def load_model(self, model_path: str, map_location: str = 'cpu') -> bool:
        """
        加载.pth格式的模型文件
        
        Args:
            model_path: 模型文件路径
            map_location: 模型加载设备位置
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(model_path):
                self.console.print(f"[red]错误: 模型文件不存在: {model_path}[/red]")
                return False
            
            # 尝试加载模型
            self.console.print(f"[yellow]正在加载模型: {model_path}[/yellow]")
            
            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=map_location)
            
            if isinstance(checkpoint, nn.Module):
                # 直接是模型对象
                self.model = checkpoint
                self.console.print("[green]✓ 成功加载模型对象[/green]")
            elif isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                    self.console.print("[green]✓ 从checkpoint中提取模型[/green]")
                elif 'state_dict' in checkpoint:
                    # 只有state_dict，需要用户提供模型结构
                    self.console.print("[yellow]警告: 只找到state_dict，需要模型结构定义[/yellow]")
                    self.model_info['state_dict'] = checkpoint['state_dict']
                    return True
                else:
                    # 假设整个dict就是state_dict
                    self.model_info['state_dict'] = checkpoint
                    self.console.print("[yellow]警告: 加载的是state_dict，需要模型结构定义[/yellow]")
                    return True
            else:
                self.console.print(f"[red]错误: 不支持的模型格式: {type(checkpoint)}[/red]")
                return False
            
            # 收集模型信息
            self._collect_model_info()
            return True
            
        except Exception as e:
            self.console.print(f"[red]加载模型时出错: {str(e)}[/red]")
            return False
    
    def _collect_model_info(self):
        """收集模型的详细信息"""
        if self.model is None:
            return
        
        self.model_info = {
            'model_class': self.model.__class__.__name__,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layers': [],
            'layer_count': 0
        }
        
        # 统计各层信息
        for name, module in self.model.named_modules():
            if name:  # 跳过根模块
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                    'input_shape': getattr(module, 'input_shape', 'Unknown'),
                    'output_shape': getattr(module, 'output_shape', 'Unknown')
                }
                self.model_info['layers'].append(layer_info)
                self.model_info['layer_count'] += 1
    
    def display_model_summary(self):
        """显示模型概要信息"""
        if not self.model_info:
            self.console.print("[red]没有加载的模型信息[/red]")
            return
        
        # 创建概要表格
        table = Table(title="模型概要信息")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="magenta")
        
        if 'model_class' in self.model_info:
            table.add_row("模型类别", self.model_info['model_class'])
        
        if 'total_params' in self.model_info:
            table.add_row("总参数数量", f"{self.model_info['total_params']:,}")
            table.add_row("可训练参数数量", f"{self.model_info['trainable_params']:,}")
        
        if 'layer_count' in self.model_info:
            table.add_row("层数", str(self.model_info['layer_count']))
        
        self.console.print(table)
    
    def display_model_architecture(self):
        """以树状结构显示模型架构"""
        if self.model is None:
            self.console.print("[red]没有加载模型[/red]")
            return
        
        # 创建架构树
        tree = Tree("🏗️ 模型架构")
        
        def add_module_to_tree(module, parent_tree, name="root"):
            """递归添加模块到树结构"""
            for child_name, child_module in module.named_children():
                # 获取模块信息
                params = sum(p.numel() for p in child_module.parameters())
                module_type = child_module.__class__.__name__
                
                # 创建节点文本
                if params > 0:
                    node_text = f"[bold blue]{child_name}[/bold blue] ({module_type}) - {params:,} params"
                else:
                    node_text = f"[bold green]{child_name}[/bold green] ({module_type})"
                
                # 添加节点
                child_tree = parent_tree.add(node_text)
                
                # 递归添加子模块
                if list(child_module.children()):
                    add_module_to_tree(child_module, child_tree, child_name)
        
        add_module_to_tree(self.model, tree)
        self.console.print(tree)
    
    def trace_model_shapes(self, input_shape: Tuple[int, ...]):
        """
        通过一次前向传播来追踪每一层的输入输出形状
        
        Args:
            input_shape: 输入数据的形状 (例如: (1, 3, 224, 224))
        """
        if self.model is None:
            self.console.print("[red]错误: 没有加载模型[/red]")
            return
            
        self.console.print(f"[yellow]正在追踪模型形状，输入形状: {input_shape}...[/yellow]")
        
        # 注册hook来捕获形状
        hooks = []
        layer_shapes = {}
        
        def get_shape_hook(name):
            def hook(module, input, output):
                input_shape = tuple(input[0].shape) if input else None
                output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
                layer_shapes[name] = {
                    'input_shape': str(input_shape),
                    'output_shape': str(output_shape)
                }
            return hook
            
        try:
            # 为每个模块注册hook
            for name, module in self.model.named_modules():
                if name:  # 跳过根模块
                    hooks.append(module.register_forward_hook(get_shape_hook(name)))
            
            # 创建虚拟输入并运行前向传播
            device = next(self.model.parameters()).device
            dummy_input = torch.zeros(input_shape).to(device)
            
            # 切换到评估模式
            training = self.model.training
            self.model.eval()
            
            with torch.no_grad():
                self.model(dummy_input)
                
            # 恢复训练模式
            self.model.train(training)
            
            # 更新model_info中的形状信息
            for layer in self.model_info.get('layers', []):
                name = layer['name']
                if name in layer_shapes:
                    layer['input_shape'] = layer_shapes[name]['input_shape']
                    layer['output_shape'] = layer_shapes[name]['output_shape']
            
            self.console.print("[green]✓ 成功捕获模型形状信息[/green]")
            
        except Exception as e:
            self.console.print(f"[red]追踪模型形状时出错: {str(e)}[/red]")
        finally:
            # 移除所有hooks
            for hook in hooks:
                hook.remove()

    def display_layer_details(self):
        """显示详细的层信息"""
        if not self.model_info.get('layers'):
            self.console.print("[red]没有层信息可显示[/red]")
            return
        
        # 创建层详情表格
        table = Table(title="层详情信息")
        table.add_column("层名称", style="cyan", no_wrap=True)
        table.add_column("类型", style="magenta")
        table.add_column("输入形状", style="blue")
        table.add_column("输出形状", style="blue")
        table.add_column("参数数量", style="yellow", justify="right")
        table.add_column("可训练参数", style="green", justify="right")
        
        for layer in self.model_info['layers']:
            table.add_row(
                layer['name'],
                layer['type'],
                str(layer.get('input_shape', 'Unknown')),
                str(layer.get('output_shape', 'Unknown')),
                f"{layer['params']:,}",
                f"{layer['trainable_params']:,}"
            )
        
        self.console.print(table)
    
    def display_state_dict_info(self):
        """显示state_dict信息（当只有权重文件时）"""
        if 'state_dict' not in self.model_info:
            self.console.print("[red]没有state_dict信息[/red]")
            return
        
        state_dict = self.model_info['state_dict']
        
        # 创建state_dict表格
        table = Table(title="State Dict 信息")
        table.add_column("参数名", style="cyan")
        table.add_column("形状", style="magenta")
        table.add_column("数据类型", style="yellow")
        table.add_column("元素数量", style="green", justify="right")
        
        total_params = 0
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                shape_str = str(list(tensor.shape))
                dtype_str = str(tensor.dtype)
                numel = tensor.numel()
                total_params += numel
                
                table.add_row(
                    key,
                    shape_str,
                    dtype_str,
                    f"{numel:,}"
                )
        
        self.console.print(table)
        self.console.print(f"\n[bold green]总参数数量: {total_params:,}[/bold green]")
    
    def get_model_info(self) -> Dict:
        """返回模型信息字典"""
        return self.model_info
    
    def export_model_info(self, output_path: str):
        """导出模型信息到文件"""
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                # 处理不能JSON序列化的对象
                exportable_info = {}
                for key, value in self.model_info.items():
                    if key != 'state_dict':  # state_dict包含tensor，不能直接序列化
                        exportable_info[key] = value
                
                json.dump(exportable_info, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]✓ 模型信息已导出到: {output_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]导出失败: {str(e)}[/red]")


# 使用示例
if __name__ == "__main__":
    viewer = ModelViewer()
    
    # 这里可以添加测试代码
    print("模型查看器初始化完成")
    print("使用方法:")
    print("viewer = ModelViewer()")
    print("viewer.load_model('your_model.pth')")
    print("viewer.display_model_summary()")
    print("viewer.display_model_architecture()")
    print("viewer.display_layer_details()")
