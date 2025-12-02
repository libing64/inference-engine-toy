# -*- coding: utf-8 -*-
"""
经典模型可视化工具 - 使用torch查看经典模型并输出SVG格式
Popular Model Viewer - View classic models using torch and export to SVG format
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Tuple
import os
import sys

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("警告: torchviz 未安装，将使用替代方法")

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("警告: graphviz 未安装，将使用替代方法")


class PopularModelViewer:
    """经典模型查看器，支持可视化并导出为SVG格式"""
    
    def __init__(self, output_dir: str = "model_visualizations"):
        """
        初始化模型查看器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        
    def load_resnet18(self, pretrained: bool = False) -> nn.Module:
        """加载ResNet18模型"""
        model = models.resnet18(pretrained=pretrained)
        model.eval()
        self.models['resnet18'] = model
        print(f"✓ 已加载 ResNet18 (pretrained={pretrained})")
        return model
    
    def load_resnet50(self, pretrained: bool = False) -> nn.Module:
        """加载ResNet50模型"""
        model = models.resnet50(pretrained=pretrained)
        model.eval()
        self.models['resnet50'] = model
        print(f"✓ 已加载 ResNet50 (pretrained={pretrained})")
        return model
    
    def load_vgg16(self, pretrained: bool = False) -> nn.Module:
        """加载VGG16模型"""
        model = models.vgg16(pretrained=pretrained)
        model.eval()
        self.models['vgg16'] = model
        print(f"✓ 已加载 VGG16 (pretrained={pretrained})")
        return model
    
    def load_vgg19(self, pretrained: bool = False) -> nn.Module:
        """加载VGG19模型"""
        model = models.vgg19(pretrained=pretrained)
        model.eval()
        self.models['vgg19'] = model
        print(f"✓ 已加载 VGG19 (pretrained={pretrained})")
        return model
    
    def load_alexnet(self, pretrained: bool = False) -> nn.Module:
        """加载AlexNet模型"""
        model = models.alexnet(pretrained=pretrained)
        model.eval()
        self.models['alexnet'] = model
        print(f"✓ 已加载 AlexNet (pretrained={pretrained})")
        return model
    
    def load_mobilenet_v2(self, pretrained: bool = False) -> nn.Module:
        """加载MobileNetV2模型"""
        model = models.mobilenet_v2(pretrained=pretrained)
        model.eval()
        self.models['mobilenet_v2'] = model
        print(f"✓ 已加载 MobileNetV2 (pretrained={pretrained})")
        return model
    
    def load_densenet121(self, pretrained: bool = False) -> nn.Module:
        """加载DenseNet121模型"""
        model = models.densenet121(pretrained=pretrained)
        model.eval()
        self.models['densenet121'] = model
        print(f"✓ 已加载 DenseNet121 (pretrained={pretrained})")
        return model
    
    def load_googlenet(self, pretrained: bool = False) -> nn.Module:
        """加载GoogLeNet模型"""
        model = models.googlenet(pretrained=pretrained)
        model.eval()
        self.models['googlenet'] = model
        print(f"✓ 已加载 GoogLeNet (pretrained={pretrained})")
        return model
    
    def load_all_popular_models(self, pretrained: bool = False):
        """加载所有经典模型"""
        print("正在加载经典模型...")
        try:
            self.load_resnet18(pretrained=pretrained)
        except Exception as e:
            print(f"加载 ResNet18 失败: {e}")
        
        try:
            self.load_resnet50(pretrained=pretrained)
        except Exception as e:
            print(f"加载 ResNet50 失败: {e}")
        
        try:
            self.load_vgg16(pretrained=pretrained)
        except Exception as e:
            print(f"加载 VGG16 失败: {e}")
        
        try:
            self.load_vgg19(pretrained=pretrained)
        except Exception as e:
            print(f"加载 VGG19 失败: {e}")
        
        try:
            self.load_alexnet(pretrained=pretrained)
        except Exception as e:
            print(f"加载 AlexNet 失败: {e}")
        
        try:
            self.load_mobilenet_v2(pretrained=pretrained)
        except Exception as e:
            print(f"加载 MobileNetV2 失败: {e}")
        
        try:
            self.load_densenet121(pretrained=pretrained)
        except Exception as e:
            print(f"加载 DenseNet121 失败: {e}")
        
        try:
            self.load_googlenet(pretrained=pretrained)
        except Exception as e:
            print(f"加载 GoogLeNet 失败: {e}")
        
        print(f"\n✓ 成功加载 {len(self.models)} 个模型")
    
    def visualize_model_to_svg(
        self, 
        model: nn.Module, 
        model_name: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        simplify: bool = True
    ) -> Optional[str]:
        """
        将模型可视化并导出为SVG格式
        
        Args:
            model: PyTorch模型
            model_name: 模型名称（用于文件名）
            input_shape: 输入形状，默认 (1, 3, 224, 224)
            simplify: 是否简化图形（减少节点数量）
            
        Returns:
            SVG文件路径，如果失败则返回None
        """
        print(f"\n正在可视化模型: {model_name}...")
        
        try:
            # 创建虚拟输入
            dummy_input = torch.randn(input_shape)
            
            # 方法1: 使用torchviz（如果可用）
            if TORCHVIZ_AVAILABLE:
                try:
                    # 运行一次前向传播
                    model.eval()
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # 创建计算图
                    dot = make_dot(output, params=dict(list(model.named_parameters())))
                    
                    # 保存为SVG
                    svg_path = os.path.join(self.output_dir, f"{model_name}.svg")
                    dot.render(
                        filename=os.path.join(self.output_dir, model_name),
                        format='svg',
                        cleanup=True
                    )
                    
                    print(f"✓ 已生成SVG: {svg_path}")
                    return svg_path
                    
                except Exception as e:
                    print(f"使用torchviz失败: {e}，尝试替代方法...")
            
            # 方法2: 使用torch.jit.trace + 手动生成SVG
            try:
                model.eval()
                traced_model = torch.jit.trace(model, dummy_input)
                
                # 获取计算图
                graph = traced_model.graph
                
                # 生成简单的SVG可视化
                svg_path = self._generate_simple_svg(graph, model_name)
                print(f"✓ 已生成SVG: {svg_path}")
                return svg_path
                
            except Exception as e:
                print(f"使用torch.jit.trace失败: {e}")
            
            # 方法3: 生成文本格式的模型结构
            svg_path = self._generate_text_based_svg(model, model_name)
            if svg_path:
                print(f"✓ 已生成文本格式SVG: {svg_path}")
                return svg_path
            
            return None
            
        except Exception as e:
            print(f"可视化模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_simple_svg(self, graph, model_name: str) -> str:
        """生成简单的SVG可视化（基于torch.jit.graph）"""
        svg_path = os.path.join(self.output_dir, f"{model_name}.svg")
        
        # 收集节点信息
        nodes = []
        edges = []
        node_id = 0
        node_map = {}
        
        for node in graph.nodes():
            node_name = node.kind()
            node_id_str = f"node_{node_id}"
            node_map[node] = node_id_str
            nodes.append((node_id_str, node_name))
            node_id += 1
            
            # 添加边
            for input_node in node.inputs():
                if input_node.node() in node_map:
                    edges.append((node_map[input_node.node()], node_id_str))
        
        # 生成SVG
        svg_content = self._create_svg_from_graph(nodes, edges, model_name)
        
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return svg_path
    
    def _create_svg_from_graph(self, nodes: List[Tuple[str, str]], edges: List[Tuple[str, str]], title: str) -> str:
        """从节点和边创建SVG内容"""
        # 简单的SVG布局（水平排列）
        node_width = 120
        node_height = 60
        spacing = 150
        start_x = 50
        start_y = 100
        
        svg_width = max(800, len(nodes) * spacing + 200)
        svg_height = 400
        
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .node {{ fill: #e1f5ff; stroke: #01579b; stroke-width: 2; }}
      .edge {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }}
      .label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
      .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#666" />
    </marker>
  </defs>
  
  <text x="{svg_width // 2}" y="30" class="title">{title}</text>
  
'''
        
        # 添加节点
        for i, (node_id, node_name) in enumerate(nodes):
            x = start_x + i * spacing
            y = start_y
            # 截断过长的节点名
            display_name = node_name[:15] + "..." if len(node_name) > 15 else node_name
            svg += f'  <rect x="{x - node_width//2}" y="{y - node_height//2}" width="{node_width}" height="{node_height}" class="node" rx="5"/>\n'
            svg += f'  <text x="{x}" y="{y + 5}" class="label">{display_name}</text>\n'
        
        # 添加边
        for i, (source, target) in enumerate(edges):
            try:
                source_idx = next(j for j, (nid, _) in enumerate(nodes) if nid == source)
                target_idx = next(j for j, (nid, _) in enumerate(nodes) if nid == target)
                
                x1 = start_x + source_idx * spacing
                y1 = start_y + node_height // 2
                x2 = start_x + target_idx * spacing
                y2 = start_y - node_height // 2
                
                svg += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge"/>\n'
            except:
                pass
        
        svg += '</svg>'
        return svg
    
    def _generate_text_based_svg(self, model: nn.Module, model_name: str) -> str:
        """生成基于文本的模型结构SVG"""
        svg_path = os.path.join(self.output_dir, f"{model_name}.svg")
        
        # 收集模型结构信息
        layers = []
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                layers.append((name, module.__class__.__name__))
        
        # 生成SVG
        svg_width = 800
        svg_height = max(400, len(layers) * 30 + 100)
        
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; text-anchor: middle; fill: #01579b; }}
      .layer-name {{ font-family: 'Courier New', monospace; font-size: 12px; fill: #333; }}
      .layer-type {{ font-family: Arial, sans-serif; font-size: 11px; fill: #666; }}
      .box {{ fill: #f5f5f5; stroke: #ddd; stroke-width: 1; }}
    </style>
  </defs>
  
  <text x="{svg_width // 2}" y="30" class="title">{model_name} - Model Architecture</text>
  
'''
        
        start_y = 70
        y_spacing = 25
        
        for i, (layer_name, layer_type) in enumerate(layers[:50]):  # 限制显示前50层
            y = start_y + i * y_spacing
            # 绘制背景框
            svg += f'  <rect x="20" y="{y - 15}" width="{svg_width - 40}" height="20" class="box" rx="3"/>\n'
            # 绘制层名
            display_name = layer_name[:60] + "..." if len(layer_name) > 60 else layer_name
            svg += f'  <text x="30" y="{y}" class="layer-name">{display_name}</text>\n'
            # 绘制层类型
            svg += f'  <text x="{svg_width - 150}" y="{y}" class="layer-type">{layer_type}</text>\n'
        
        if len(layers) > 50:
            svg += f'  <text x="{svg_width // 2}" y="{start_y + 50 * y_spacing + 20}" class="layer-type">... 还有 {len(layers) - 50} 层未显示</text>\n'
        
        svg += '</svg>'
        
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg)
        
        return svg_path
    
    def visualize_all_models(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """可视化所有已加载的模型"""
        print(f"\n开始可视化 {len(self.models)} 个模型...")
        print(f"输出目录: {self.output_dir}")
        
        results = {}
        for model_name, model in self.models.items():
            svg_path = self.visualize_model_to_svg(
                model, 
                model_name, 
                input_shape=input_shape
            )
            results[model_name] = svg_path
        
        print(f"\n✓ 完成！已生成 {sum(1 for v in results.values() if v)} 个SVG文件")
        return results
    
    def get_model_info(self, model_name: str) -> dict:
        """获取模型信息"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'name': model_name,
            'class': model.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model': model
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="经典模型可视化工具")
    parser.add_argument('--models', nargs='+', 
                       choices=['resnet18', 'resnet50', 'vgg16', 'vgg19', 'alexnet', 
                               'mobilenet_v2', 'densenet121', 'googlenet', 'all'],
                       default=['all'],
                       help='要可视化的模型列表')
    parser.add_argument('--pretrained', action='store_true',
                       help='是否使用预训练权重')
    parser.add_argument('--output-dir', type=str, default='model_visualizations',
                       help='输出目录')
    parser.add_argument('--input-shape', type=int, nargs=4, 
                       default=[1, 3, 224, 224],
                       help='输入形状 (batch, channels, height, width)')
    
    args = parser.parse_args()
    
    # 创建查看器
    viewer = PopularModelViewer(output_dir=args.output_dir)
    
    # 加载模型
    if 'all' in args.models:
        viewer.load_all_popular_models(pretrained=args.pretrained)
    else:
        for model_name in args.models:
            try:
                if model_name == 'resnet18':
                    viewer.load_resnet18(pretrained=args.pretrained)
                elif model_name == 'resnet50':
                    viewer.load_resnet50(pretrained=args.pretrained)
                elif model_name == 'vgg16':
                    viewer.load_vgg16(pretrained=args.pretrained)
                elif model_name == 'vgg19':
                    viewer.load_vgg19(pretrained=args.pretrained)
                elif model_name == 'alexnet':
                    viewer.load_alexnet(pretrained=args.pretrained)
                elif model_name == 'mobilenet_v2':
                    viewer.load_mobilenet_v2(pretrained=args.pretrained)
                elif model_name == 'densenet121':
                    viewer.load_densenet121(pretrained=args.pretrained)
                elif model_name == 'googlenet':
                    viewer.load_googlenet(pretrained=args.pretrained)
            except Exception as e:
                print(f"加载 {model_name} 失败: {e}")
    
    # 可视化所有模型
    input_shape = tuple(args.input_shape)
    results = viewer.visualize_all_models(input_shape=input_shape)
    
    # 打印结果
    print("\n生成的SVG文件:")
    for model_name, svg_path in results.items():
        if svg_path:
            print(f"  ✓ {model_name}: {svg_path}")
        else:
            print(f"  ✗ {model_name}: 生成失败")


if __name__ == "__main__":
    main()

