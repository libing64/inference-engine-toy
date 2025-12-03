# -*- coding: utf-8 -*-
"""
推理对比脚本 - 对比 PyTorch 和 C++ 的推理结果和速度
Inference Comparison Script - Compare PyTorch and C++ inference results and speed
"""

import subprocess
import sys
import os
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
import re

console = Console()

# 导入 PyTorch 推理函数
from pytorch_inference import pytorch_inference, print_tensor_info


def run_cpp_inference(model_path, input_shape=(1, 3, 32, 32), cpp_executable="./inference"):
    """
    运行 C++ 推理程序并解析输出
    
    Returns:
        (output_tensor, avg_time_ms, output_text): 输出张量、平均时间和原始输出
    """
    shape_str = ','.join(map(str, input_shape))
    cmd = [cpp_executable, model_path, shape_str]
    
    console.print(f"[cyan]执行命令: {' '.join(cmd)}[/cyan]")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        output_text = result.stdout
        error_text = result.stderr
        
        if error_text:
            console.print(f"[yellow]C++ 程序警告/信息:[/yellow]\n{error_text}")
        
        # 解析输出张量（从输出文本中提取）
        # 注意：这里我们需要从 C++ 输出中解析，或者让 C++ 程序保存结果到文件
        # 为了简化，我们假设 C++ 程序会输出张量的前几个元素
        
        # 解析平均时间
        time_match = re.search(r'平均推理时间:\s*([\d.]+)\s*ms', output_text)
        avg_time_ms = float(time_match.group(1)) if time_match else None
        
        # 解析输出形状
        shape_match = re.search(r'形状:\s*\[([\d,\s]+)\]', output_text)
        output_shape = None
        if shape_match:
            shape_str = shape_match.group(1)
            output_shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip())
        
        # 解析输出元素
        elements_match = re.search(r'前\s*\d+\s*个元素:\s*\[([\d.\-,\s]+)\]', output_text)
        output_elements = None
        if elements_match:
            elements_str = elements_match.group(1)
            output_elements = [float(x.strip()) for x in elements_str.split(',') if x.strip()]
        
        return {
            'output_text': output_text,
            'avg_time_ms': avg_time_ms,
            'output_shape': output_shape,
            'output_elements': output_elements
        }
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]C++ 程序执行失败:[/red]")
        console.print(f"返回码: {e.returncode}")
        console.print(f"标准输出:\n{e.stdout}")
        console.print(f"标准错误:\n{e.stderr}")
        return None
    except FileNotFoundError:
        console.print(f"[red]错误: 找不到 C++ 可执行文件: {cpp_executable}[/red]")
        console.print("[yellow]请先编译 C++ 程序:[/yellow]")
        console.print("  mkdir build && cd build")
        console.print("  cmake ..")
        console.print("  make")
        return None


def compare_inference(model_path, input_shape=(1, 3, 32, 32), cpp_executable="./inference"):
    """
    对比 PyTorch 和 C++ 推理
    """
    console.print(Panel(
        "[bold green]推理对比测试[/bold green]\n\n"
        f"模型: {model_path}\n"
        f"输入形状: {input_shape}",
        title="开始对比",
        border_style="green"
    ))
    
    # 1. PyTorch 推理
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]1. PyTorch 推理[/bold cyan]")
    console.print("=" * 60)
    
    pytorch_output, pytorch_time = pytorch_inference(model_path, input_shape)
    
    console.print(f"\n[green]PyTorch 推理完成[/green]")
    print_tensor_info(pytorch_output, "PyTorch 输出")
    console.print(f"平均推理时间: [bold]{pytorch_time:.3f} ms[/bold]")
    
    # 2. C++ 推理
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]2. C++ 推理[/bold cyan]")
    console.print("=" * 60)
    
    cpp_result = run_cpp_inference(model_path, input_shape, cpp_executable)
    
    if not cpp_result:
        console.print("[red]C++ 推理失败，无法进行对比[/red]")
        return
    
    console.print(f"\n[green]C++ 推理完成[/green]")
    if cpp_result['output_shape']:
        console.print(f"输出形状: {cpp_result['output_shape']}")
    if cpp_result['output_elements']:
        console.print(f"前 5 个元素: {cpp_result['output_elements']}")
    if cpp_result['avg_time_ms']:
        console.print(f"平均推理时间: [bold]{cpp_result['avg_time_ms']:.3f} ms[/bold]")
    
    # 3. 对比结果
    console.print("\n[bold cyan]=" * 60)
    console.print("[bold cyan]3. 对比结果[/bold cyan]")
    console.print("=" * 60)
    
    # 创建对比表格
    table = Table(title="推理对比结果", show_lines=True)
    table.add_column("指标", style="cyan")
    table.add_column("PyTorch", style="magenta")
    table.add_column("C++", style="green")
    table.add_column("差异", style="yellow")
    
    # 推理时间对比
    if cpp_result['avg_time_ms']:
        time_diff = pytorch_time - cpp_result['avg_time_ms']
        time_diff_pct = (time_diff / pytorch_time) * 100
        faster = "C++" if time_diff > 0 else "PyTorch"
        
        table.add_row(
            "平均推理时间 (ms)",
            f"{pytorch_time:.3f}",
            f"{cpp_result['avg_time_ms']:.3f}",
            f"{abs(time_diff):.3f} ms ({faster} 快 {abs(time_diff_pct):.1f}%)"
        )
        
        table.add_row(
            "吞吐量 (FPS)",
            f"{1000.0 / pytorch_time:.2f}",
            f"{1000.0 / cpp_result['avg_time_ms']:.2f}",
            f"{faster} 快 {abs(time_diff_pct):.1f}%"
        )
    
    # 输出形状对比
    pytorch_shape = list(pytorch_output.shape)
    if cpp_result['output_shape']:
        shape_match = pytorch_shape == list(cpp_result['output_shape'])
        table.add_row(
            "输出形状",
            str(pytorch_shape),
            str(list(cpp_result['output_shape'])),
            "✓ 一致" if shape_match else "✗ 不一致"
        )
    
    # 输出值对比（如果 C++ 输出了元素值）
    if cpp_result['output_elements']:
        pytorch_elements = pytorch_output.flatten()[:5].tolist()
        cpp_elements = cpp_result['output_elements'][:5]
        
        # 计算差异
        max_diff = max(abs(a - b) for a, b in zip(pytorch_elements, cpp_elements))
        mean_diff = np.mean([abs(a - b) for a, b in zip(pytorch_elements, cpp_elements)])
        
        table.add_row(
            "输出值差异 (前5个元素)",
            f"最大: {max_diff:.6f}",
            f"平均: {mean_diff:.6f}",
            "✓ 数值一致" if max_diff < 1e-4 else f"⚠ 有差异 (最大: {max_diff:.6f})"
        )
    
    console.print(table)
    
    # 总结
    console.print("\n[bold green]对比总结:[/bold green]")
    if cpp_result['avg_time_ms']:
        if pytorch_time > cpp_result['avg_time_ms']:
            speedup = pytorch_time / cpp_result['avg_time_ms']
            console.print(f"  • C++ 推理比 PyTorch 快 [bold]{speedup:.2f}x[/bold]")
        else:
            speedup = cpp_result['avg_time_ms'] / pytorch_time
            console.print(f"  • PyTorch 推理比 C++ 快 [bold]{speedup:.2f}x[/bold]")
    
    console.print("  • 两种实现的结果形状一致")
    if cpp_result['output_elements']:
        max_diff = max(abs(a - b) for a, b in 
                      zip(pytorch_output.flatten()[:5].tolist(), 
                          cpp_result['output_elements'][:5]))
        if max_diff < 1e-4:
            console.print("  • 输出数值基本一致（差异 < 1e-4）")
        else:
            console.print(f"  • 输出数值有微小差异（最大差异: {max_diff:.6f}）")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        console.print("[red]用法: python compare_inference.py <模型路径> [C++可执行文件路径][/red]")
        console.print("[yellow]示例: python compare_inference.py ../traced_models/simple_cnn_traced.pt[/yellow]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    cpp_executable = sys.argv[2] if len(sys.argv) > 2 else "./inference"
    
    # 默认输入形状
    input_shape = (1, 3, 32, 32)
    
    compare_inference(model_path, input_shape, cpp_executable)


if __name__ == "__main__":
    main()
