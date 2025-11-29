# -*- coding: utf-8 -*-
"""
Web 界面 - 基于 Streamlit 的模型查看器和推理引擎
Web Interface - Streamlit-based model viewer and inference engine
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import io
from contextlib import redirect_stdout

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_viewer import ModelViewer
from inference_engine import InferenceEngine

# 尝试导入示例模型类，以支持反序列化
try:
    from examples import SimpleCNN, SimpleMLP, SimpleResNet, BasicBlock
except ImportError:
    pass

st.set_page_config(
    page_title="模型查看器 & 推理引擎",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧠 PyTorch 模型查看器 & 推理引擎")
    st.markdown("---")

    # --- Sidebar: 模型加载 ---
    st.sidebar.header("📂 模型加载")
    
    # 选项：上传文件 或 使用示例
    upload_option = st.sidebar.radio("选择模型来源", ["上传模型文件", "使用示例模型"])
    
    model_path = None
    uploaded_file = None
    
    if upload_option == "上传模型文件":
        uploaded_file = st.sidebar.file_uploader("上传 .pth/.pt 文件", type=['pth', 'pt'])
        if uploaded_file:
            # 保存临时文件以便 ModelViewer 加载
            # 获取原文件后缀
            ext = os.path.splitext(uploaded_file.name)[1]
            temp_filename = f"temp_model{ext}"
            
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            model_path = temp_filename
            
    else:
        # 列出 example_models 目录下的文件
        example_dir = "example_models"
        if os.path.exists(example_dir):
            files = [f for f in os.listdir(example_dir) if f.endswith('.pth') or f.endswith('.pt')]
            selected_file = st.sidebar.selectbox("选择示例模型", files)
            if selected_file:
                model_path = os.path.join(example_dir, selected_file)
        else:
            st.sidebar.warning("未找到示例模型目录，请先运行 examples.py 生成示例。")

    # --- 主逻辑 ---
    if model_path:
        try:
            # 初始化查看器
            viewer = ModelViewer()
            
            # 加载模型
            # 捕获标准输出以隐藏 rich 的打印信息，或者我们不调用 display 方法
            if viewer.load_model(model_path):
                st.sidebar.success(f"成功加载模型: {os.path.basename(model_path)}")
                
                # 获取模型信息
                info = viewer.get_model_info()
                
                # 创建 Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 模型概览", "📋 层详情", "🏗️ 架构结构", "⚡ 推理实验室"])
                
                # --- Tab 1: 模型概览 ---
                with tab1:
                    st.subheader("模型基本信息")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("模型类别", info.get('model_class', 'Unknown'))
                    with col2:
                        st.metric("总参数量", f"{info.get('total_params', 0):,}")
                    with col3:
                        st.metric("可训练参数", f"{info.get('trainable_params', 0):,}")
                    
                    st.metric("层数", info.get('layer_count', 0))
                    
                    # 如果有 state_dict 信息
                    if 'state_dict' in info:
                        st.info("⚠️ 注意：加载的是 State Dict (权重字典)，部分结构信息可能不完整。")

                # --- Tab 2: 层详情 ---
                with tab2:
                    st.subheader("层详细信息")
                    
                    layers = info.get('layers', [])
                    if layers:
                        # 转换为 DataFrame 展示
                        df = pd.DataFrame(layers)
                        # 重命名列以更友好显示
                        df = df.rename(columns={
                            'name': '层名称',
                            'type': '类型',
                            'input_shape': '输入形状',
                            'output_shape': '输出形状',
                            'params': '参数数量',
                            'trainable_params': '可训练参数'
                        })
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("暂无层信息")

                # --- Tab 3: 架构结构 ---
                with tab3:
                    st.subheader("模型架构树")
                    # 由于 ModelViewer 使用 rich 打印树，我们需要一种方法获取树的文本表示
                    # 这里我们简单地重构一个递归函数来生成文本树
                    
                    if viewer.model:
                        tree_text = get_model_tree_text(viewer.model)
                        st.text(tree_text)
                    else:
                        st.text("仅有权重信息，无法显示完整架构树")

                # --- Tab 4: 推理实验室 ---
                with tab4:
                    st.subheader("模型推理与形状追踪")
                    
                    st.markdown("""
                    在此处输入数据的形状，执行一次推理。这将：
                    1. 验证模型是否能处理该形状的输入
                    2. **自动更新** "层详情" 中的输入/输出形状信息
                    3. 显示逐层的推理耗时
                    """)
                    
                    # 输入形状
                    default_shape = "1, 3, 224, 224"
                    shape_input = st.text_input("输入数据形状 (逗号分隔)", value=default_shape)
                    
                    if st.button("执行推理 / 追踪形状"):
                        try:
                            # 解析形状
                            shape_list = [int(x.strip()) for x in shape_input.split(',')]
                            input_shape = tuple(shape_list)
                            
                            # 1. 更新形状信息 (调用 ModelViewer 的 trace 功能)
                            with st.spinner("正在追踪形状信息..."):
                                # 捕获输出防止干扰
                                f = io.StringIO()
                                with redirect_stdout(f):
                                    viewer.trace_model_shapes(input_shape)
                                
                                # 强制刷新 Tab 2 的显示需要重新获取 info
                                # 但由于 info 是引用，ModelViewer 内部修改后这里应该能看到更新
                                st.success("形状追踪完成！请查看 '层详情' 标签页更新后的形状信息。")
                            
                            # 2. 执行推理引擎 (展示详细步骤)
                            if hasattr(viewer, 'model') and viewer.model:
                                engine = InferenceEngine(verbose=False) # 关闭 verbose，我们自己显示
                                engine.load_model(viewer.model)
                                
                                # 创建随机输入
                                device = next(viewer.model.parameters()).device
                                input_data = torch.randn(*input_shape).to(device)
                                
                                with st.spinner("正在执行推理..."):
                                    output = engine.infer(input_data, detailed=False)
                                    steps = engine.get_inference_steps()
                                
                                st.success("推理成功！")
                                st.write(f"**输出张量形状:** `{tuple(output.shape)}`")
                                
                                # 显示推理步骤表格
                                st.subheader("推理步骤详解")
                                step_data = []
                                for step in steps:
                                    step_data.append({
                                        "层名称": step.layer_name,
                                        "操作": step.operation,
                                        "输入": str(step.input_shape),
                                        "输出": str(step.output_shape),
                                        "耗时 (ms)": f"{step.execution_time * 1000:.4f}"
                                    })
                                st.dataframe(pd.DataFrame(step_data), use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"执行失败: {str(e)}")
                            
            else:
                st.error("模型加载失败，请检查文件格式。")
                
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.exception(e)

def get_model_tree_text(model):
    """生成模型结构的文本树表示"""
    lines = []
    
    def _add_layer(module, prefix="", is_last=True):
        # 获取子模块
        children = list(module.named_children())
        
        for i, (name, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            connector = "└── " if is_last_child else "├── "
            
            # 获取层信息
            params = sum(p.numel() for p in child.parameters())
            type_name = child.__class__.__name__
            info = f"{type_name}"
            if params > 0:
                info += f" [Params: {params:,}]"
            
            lines.append(f"{prefix}{connector}{name} ({info})")
            
            # 递归
            new_prefix = prefix + ("    " if is_last_child else "│   ")
            _add_layer(child, new_prefix, is_last_child)
            
    lines.append(f"root ({model.__class__.__name__})")
    _add_layer(model)
    return "\n".join(lines)

if __name__ == "__main__":
    main()
