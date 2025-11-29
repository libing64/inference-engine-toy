# -*- coding: utf-8 -*-
"""
推理引擎 - 一个小型的可读性优先的模型推理引擎
Inference Engine - A small, readability-focused model inference engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
import time
import numpy as np


class InferenceStep:
    """推理步骤记录"""
    
    def __init__(self, layer_name: str, operation: str, input_shape: tuple, 
                 output_shape: tuple, execution_time: float, parameters: Dict = None):
        self.layer_name = layer_name
        self.operation = operation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.execution_time = execution_time
        self.parameters = parameters or {}
        self.timestamp = time.time()


class InferenceEngine:
    """简易推理引擎，优先保证可读性和过程可视化"""
    
    def __init__(self, verbose: bool = True, track_intermediate: bool = True):
        self.console = Console()
        self.verbose = verbose
        self.track_intermediate = track_intermediate
        
        # 推理历史记录
        self.inference_steps: List[InferenceStep] = []
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}
        
        # 支持的操作映射
        self.operation_handlers = {
            'Linear': self._handle_linear,
            'Conv2d': self._handle_conv2d,
            'Conv1d': self._handle_conv1d,
            'BatchNorm2d': self._handle_batchnorm2d,
            'BatchNorm1d': self._handle_batchnorm1d,
            'ReLU': self._handle_relu,
            'Sigmoid': self._handle_sigmoid,
            'Tanh': self._handle_tanh,
            'Softmax': self._handle_softmax,
            'MaxPool2d': self._handle_maxpool2d,
            'AdaptiveAvgPool2d': self._handle_adaptive_avgpool2d,
            'Flatten': self._handle_flatten,
            'Dropout': self._handle_dropout,
        }
    
    def load_model(self, model: nn.Module):
        """加载要推理的模型"""
        self.model = model
        self.model.eval()  # 设置为评估模式
        
        if self.verbose:
            self.console.print("[green]✓ 模型已加载并设置为评估模式[/green]")
    
    def infer(self, input_data: torch.Tensor, detailed: bool = True) -> torch.Tensor:
        """
        执行模型推理
        
        Args:
            input_data: 输入数据
            detailed: 是否显示详细的推理过程
            
        Returns:
            推理结果
        """
        if not hasattr(self, 'model'):
            raise ValueError("请先使用load_model()加载模型")
        
        # 清空之前的推理记录
        self.inference_steps.clear()
        self.intermediate_outputs.clear()
        
        if self.verbose:
            self.console.print(f"[yellow]开始推理，输入形状: {list(input_data.shape)}[/yellow]")
        
        start_time = time.time()
        
        # 注册hooks
        hooks = []
        
        def get_layer_hook(name, layer):
            def hook(module, input, output):
                # 记录时间（估算）
                execution_time = 0.0 # hook中很难准确测量计算时间，因为是异步的
                
                input_shape = tuple(input[0].shape) if input else None
                output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
                
                # 获取参数信息
                parameters = self._get_layer_parameters(module)
                
                step = InferenceStep(
                    layer_name=name,
                    operation=module.__class__.__name__,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    execution_time=execution_time, # 暂时设为0
                    parameters=parameters
                )
                self.inference_steps.append(step)
                
                # 保存中间输出
                if self.track_intermediate and isinstance(output, torch.Tensor):
                    self.intermediate_outputs[name] = output.detach().clone()
                
                # 实时显示（如果在verbose模式）
                if self.verbose:
                    self._display_step_info(step)
                    
            return hook

        try:
            # 只为叶子节点注册hook，避免重复记录
            for name, layer in self.model.named_modules():
                if name and len(list(layer.children())) == 0:  # 是叶子节点
                    hooks.append(layer.register_forward_hook(get_layer_hook(name, layer)))
            
            # 执行推理
            with torch.no_grad():
                current_output = self.model(input_data)
                
        except Exception as e:
            self.console.print(f"[red]推理过程出错: {str(e)}[/red]")
            raise e
        finally:
            # 移除所有hooks
            for hook in hooks:
                hook.remove()
        
        total_time = time.time() - start_time
        
        if detailed and self.verbose:
            self._display_inference_summary(total_time)
        
        return current_output

    def _execute_layer(self, layer_name: str, layer: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """(已弃用) 执行单个层的推理"""
        # 保留此方法以兼容旧代码，但不再主要使用
        return layer(input_tensor)
    
    def _get_layer_parameters(self, layer: nn.Module) -> Dict:
        """获取层的参数信息"""
        parameters = {}
        
        # 通用参数
        if hasattr(layer, 'weight') and layer.weight is not None:
            parameters['weight_shape'] = tuple(layer.weight.shape)
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            parameters['bias_shape'] = tuple(layer.bias.shape)
        
        # 特定层的参数
        layer_type = layer.__class__.__name__
        
        if layer_type in ['Conv2d', 'Conv1d']:
            parameters.update({
                'kernel_size': getattr(layer, 'kernel_size', None),
                'stride': getattr(layer, 'stride', None),
                'padding': getattr(layer, 'padding', None),
                'dilation': getattr(layer, 'dilation', None),
            })
        
        elif layer_type in ['MaxPool2d', 'AvgPool2d']:
            parameters.update({
                'kernel_size': getattr(layer, 'kernel_size', None),
                'stride': getattr(layer, 'stride', None),
                'padding': getattr(layer, 'padding', None),
            })
        
        elif layer_type == 'Dropout':
            parameters['p'] = getattr(layer, 'p', None)
        
        elif layer_type == 'Softmax':
            parameters['dim'] = getattr(layer, 'dim', None)
        
        return parameters
    
    def _display_step_info(self, step: InferenceStep):
        """显示单个推理步骤的信息"""
        input_str = f"{step.input_shape}"
        output_str = f"{step.output_shape}"
        time_str = f"{step.execution_time*1000:.2f}ms"
        
        self.console.print(f"  [cyan]{step.layer_name}[/cyan] ({step.operation}): "
                          f"{input_str} → {output_str} [{time_str}]")
    
    def _display_inference_summary(self, total_time: float):
        """显示推理总结"""
        self.console.print("\n" + "="*50)
        self.console.print(f"[bold green]推理完成! 总耗时: {total_time*1000:.2f}ms[/bold green]")
        
        # 创建详细统计表
        table = Table(title="推理步骤统计")
        table.add_column("层名称", style="cyan", no_wrap=True)
        table.add_column("操作类型", style="magenta")
        table.add_column("输入形状", style="yellow")
        table.add_column("输出形状", style="green")
        table.add_column("耗时(ms)", style="red", justify="right")
        
        for step in self.inference_steps:
            table.add_row(
                step.layer_name,
                step.operation,
                str(step.input_shape),
                str(step.output_shape),
                f"{step.execution_time*1000:.2f}"
            )
        
        self.console.print(table)
    
    # 具体的操作处理函数
    def _handle_linear(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """处理全连接层"""
        return F.linear(x, layer.weight, layer.bias)
    
    def _handle_conv2d(self, layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """处理2D卷积层"""
        return F.conv2d(x, layer.weight, layer.bias, 
                       layer.stride, layer.padding, layer.dilation, layer.groups)
    
    def _handle_conv1d(self, layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """处理1D卷积层"""
        return F.conv1d(x, layer.weight, layer.bias,
                       layer.stride, layer.padding, layer.dilation, layer.groups)
    
    def _handle_batchnorm2d(self, layer: nn.BatchNorm2d, x: torch.Tensor) -> torch.Tensor:
        """处理2D批归一化层"""
        return F.batch_norm(x, layer.running_mean, layer.running_var,
                          layer.weight, layer.bias, False, layer.momentum, layer.eps)
    
    def _handle_batchnorm1d(self, layer: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        """处理1D批归一化层"""
        return F.batch_norm(x, layer.running_mean, layer.running_var,
                          layer.weight, layer.bias, False, layer.momentum, layer.eps)
    
    def _handle_relu(self, layer: nn.ReLU, x: torch.Tensor) -> torch.Tensor:
        """处理ReLU激活函数"""
        return F.relu(x, inplace=layer.inplace if hasattr(layer, 'inplace') else False)
    
    def _handle_sigmoid(self, layer: nn.Sigmoid, x: torch.Tensor) -> torch.Tensor:
        """处理Sigmoid激活函数"""
        return torch.sigmoid(x)
    
    def _handle_tanh(self, layer: nn.Tanh, x: torch.Tensor) -> torch.Tensor:
        """处理Tanh激活函数"""
        return torch.tanh(x)
    
    def _handle_softmax(self, layer: nn.Softmax, x: torch.Tensor) -> torch.Tensor:
        """处理Softmax激活函数"""
        return F.softmax(x, dim=layer.dim)
    
    def _handle_maxpool2d(self, layer: nn.MaxPool2d, x: torch.Tensor) -> torch.Tensor:
        """处理2D最大池化层"""
        return F.max_pool2d(x, layer.kernel_size, layer.stride, 
                           layer.padding, layer.dilation, layer.ceil_mode, layer.return_indices)
    
    def _handle_adaptive_avgpool2d(self, layer: nn.AdaptiveAvgPool2d, x: torch.Tensor) -> torch.Tensor:
        """处理自适应平均池化层"""
        return F.adaptive_avg_pool2d(x, layer.output_size)
    
    def _handle_flatten(self, layer: nn.Flatten, x: torch.Tensor) -> torch.Tensor:
        """处理展平层"""
        return torch.flatten(x, layer.start_dim, layer.end_dim)
    
    def _handle_dropout(self, layer: nn.Dropout, x: torch.Tensor) -> torch.Tensor:
        """处理Dropout层"""
        return F.dropout(x, p=layer.p, training=False)  # 推理时不使用dropout
    
    def get_intermediate_output(self, layer_name: str) -> Optional[torch.Tensor]:
        """获取指定层的中间输出"""
        return self.intermediate_outputs.get(layer_name)
    
    def get_inference_steps(self) -> List[InferenceStep]:
        """获取推理步骤记录"""
        return self.inference_steps
    
    def benchmark_model(self, input_data: torch.Tensor, num_runs: int = 10) -> Dict:
        """对模型进行基准测试"""
        if not hasattr(self, 'model'):
            raise ValueError("请先使用load_model()加载模型")
        
        self.console.print(f"[yellow]开始基准测试，运行 {num_runs} 次...[/yellow]")
        
        times = []
        
        with Progress() as progress:
            task = progress.add_task("基准测试进度", total=num_runs)
            
            for i in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(input_data)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                progress.update(task, advance=1)
        
        # 计算统计信息
        times = np.array(times)
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'total_runs': num_runs
        }
        
        # 显示结果
        self._display_benchmark_results(stats)
        
        return stats
    
    def _display_benchmark_results(self, stats: Dict):
        """显示基准测试结果"""
        table = Table(title="基准测试结果")
        table.add_column("指标", style="cyan")
        table.add_column("时间(ms)", style="magenta", justify="right")
        
        table.add_row("平均时间", f"{stats['mean_time']*1000:.2f}")
        table.add_row("标准差", f"{stats['std_time']*1000:.2f}")
        table.add_row("最小时间", f"{stats['min_time']*1000:.2f}")
        table.add_row("最大时间", f"{stats['max_time']*1000:.2f}")
        table.add_row("中位数时间", f"{stats['median_time']*1000:.2f}")
        table.add_row("测试次数", str(stats['total_runs']))
        
        self.console.print(table)


# 使用示例
if __name__ == "__main__":
    engine = InferenceEngine(verbose=True)
    
    print("推理引擎初始化完成")
    print("使用方法:")
    print("engine = InferenceEngine()")
    print("engine.load_model(your_model)")
    print("result = engine.infer(input_data)")
    print("stats = engine.benchmark_model(input_data)")
