# -*- coding: utf-8 -*-
"""
经典模型可视化工具 - 使用torch查看经典模型并输出HTML格式
Popular Model Viewer - View classic models using torch and export to HTML format
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Tuple
import os
import sys
from torchvista import trace_model





class PopularModelViewer:
    """经典模型查看器，支持可视化并导出为HTML格式"""
    
    # 经典模型的默认输入尺寸映射 (batch, channels, height, width)
    MODEL_INPUT_SIZES = {
        'resnet18': (1, 3, 224, 224),
        'resnet50': (1, 3, 224, 224),
        'vgg16': (1, 3, 224, 224),
        'vgg19': (1, 3, 224, 224),
        'alexnet': (1, 3, 224, 224),  # PyTorch torchvision 使用 224x224
        'mobilenet_v2': (1, 3, 224, 224),
        'densenet121': (1, 3, 224, 224),
        'googlenet': (1, 3, 224, 224),
    }
    
    def __init__(self, output_dir: str = "model_visualizations"):
        """
        初始化模型查看器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
    
    def get_model_input_size(self, model_name: str) -> Tuple[int, ...]:
        """
        获取模型的默认输入尺寸
        
        Args:
            model_name: 模型名称
            
        Returns:
            输入尺寸元组 (batch, channels, height, width)
        """
        return self.MODEL_INPUT_SIZES.get(model_name, (1, 3, 224, 224))
        
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
    
    def visualize_model_to_html(
        self, 
        model: nn.Module, 
        model_name: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        simplify: bool = True
    ) -> Optional[str]:
        """
        将模型可视化并导出为HTML格式
        
        Args:
            model: PyTorch模型
            model_name: 模型名称（用于文件名）
            input_shape: 输入形状，如果为None则使用模型默认尺寸
            simplify: 是否简化图形（减少节点数量）
            
        Returns:
            HTML文件路径，如果失败则返回None
        """
        print(f"\n正在可视化模型: {model_name}...")
        
        try:
            # 如果没有指定输入尺寸，使用模型默认尺寸
            if input_shape is None:
                input_shape = self.get_model_input_size(model_name)
                print(f"  使用默认输入尺寸: {input_shape}")
            else:
                print(f"  使用指定输入尺寸: {input_shape}")
            
            # 创建虚拟输入
            # input shape: (batch, channels, height, width)
            print(f"input_shape: {input_shape}")
            dummy_input = torch.randn(input_shape)
            
            # 使用torchvista的trace_model追踪模型
            # forced_module_tracing_depth: 强制追踪模块的深度
            # collapse_modules_after_depth: 在指定深度后折叠模块
            # show_non_gradient_nodes: 是否显示非梯度节点
            # export_format: 导出格式 ('svg', 'png', 'html'等)
            trace_model(
                model, 
                dummy_input, 
                forced_module_tracing_depth=3, 
                collapse_modules_after_depth=1, 
                show_non_gradient_nodes=False, 
                export_format='html'
            )
            return
        except Exception as e:
            print(f"可视化模型 {model_name} 失败: {e}")
            return None
            
    
    
    def visualize_all_models(self, input_shape: Optional[Tuple[int, ...]] = None):
        """
        可视化所有已加载的模型
        
        Args:
            input_shape: 输入形状，如果为None则每个模型使用其默认尺寸
        """
        print(f"\n开始可视化 {len(self.models)} 个模型...")
        print(f"输出目录: {self.output_dir}")
        
        if input_shape is None:
            print("使用每个模型的默认输入尺寸")
        else:
            print(f"所有模型使用统一输入尺寸: {input_shape}")
        
        results = {}
        for model_name, model in self.models.items():
            # 如果指定了统一的输入尺寸，使用它；否则使用模型默认尺寸
            model_input_shape = input_shape if input_shape is not None else None
            html_path = self.visualize_model_to_html(
                model, 
                model_name, 
                input_shape=model_input_shape
            )
            results[model_name] = html_path
        
        print(f"\n✓ 完成！已生成 {sum(1 for v in results.values() if v)} 个HTML文件")
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
                       default=None,
                       help='输入形状 (batch, channels, height, width)，如果不指定则使用每个模型的默认尺寸')
    
    args = parser.parse_args()
    
    # 创建查看器
    viewer = PopularModelViewer(output_dir=args.output_dir)
    
    # 打印模型输入尺寸信息
    print("=" * 60)
    print("经典模型的默认输入尺寸:")
    print("=" * 60)
    for model_name, size in viewer.MODEL_INPUT_SIZES.items():
        print(f"  {model_name:15s}: {size}")
    print("=" * 60)
    print()
    
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
    input_shape = tuple(args.input_shape) if args.input_shape else None
    results = viewer.visualize_all_models(input_shape=input_shape)
    
    # 打印结果
    print("\n生成的HTML文件:")
    for model_name, html_path in results.items():
        if html_path:
            print(f"  ✓ {model_name}: {html_path}")
        else:
            print(f"  ✗ {model_name}: 生成失败")


if __name__ == "__main__":
    main()

