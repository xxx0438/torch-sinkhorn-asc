import gradio as gr
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 导入你的核心代码
sys.path.append("src")
from sinkhorn import SinkhornLayer
from scheduler import EPH_ASC_Scheduler

def simulate(n_points, decay_rate, safety_slope):
    """
    运行对比模拟：标准退火 vs EPH-ASC
    """
    # 模拟设置
    batch_size = 1
    n_epochs = 100
    n_points = int(n_points)
    
    # 初始化
    sinkhorn = SinkhornLayer(n_iters=20)
    
    # 1. 标准指数退火 (Standard Exponential)
    eps_std = 1.0
    history_std = []
    
    # 2. 你的算法 (EPH-ASC)
    scheduler = EPH_ASC_Scheduler(
        init_epsilon=1.0, 
        min_epsilon=0.01, 
        decay_rate=decay_rate, 
        k_safe=safety_slope
    )
    history_asc = []
    eps_asc_log = []

    # 固定随机种子以便对比
    torch.manual_seed(42)
    
    # 模拟数据流
    cost_matrices = [torch.randn(batch_size, n_points, n_points) for _ in range(n_epochs)]

    for epoch in range(n_epochs):
        C = cost_matrices[epoch]
        
        # --- 运行标准退火 ---
        P_std = sinkhorn(C, eps_std)
        # 简单的熵计算 (Entropy) 作为监控指标
        entropy_std = -(P_std * (P_std + 1e-8).log()).sum(dim=-1).mean().item()
        history_std.append(entropy_std)
        # 标准更新：盲目降温
        eps_std = max(0.01, eps_std * decay_rate)
        
        # --- 运行 EPH-ASC ---
        curr_eps = scheduler.epsilon
        P_asc = sinkhorn(C, curr_eps)
        entropy_asc = -(P_asc * (P_asc + 1e-8).log()).sum(dim=-1).mean().item()
        history_asc.append(entropy_asc)
        eps_asc_log.append(curr_eps)
        
        # 自适应更新
        scheduler.step(P_asc)

    # --- 画图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: 熵 (Entropy) - 显示是否过早坍缩
    ax1.plot(history_std, label="Standard Annealing", linestyle="--", color="blue")
    ax1.plot(history_asc, label="EPH-ASC (Ours)", color="red", linewidth=2)
    ax1.set_title("Plan Entropy (Uncertainty)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 温度变化 - 显示“刹车”机制
    ax2.plot([decay_rate**i for i in range(n_epochs)], label="Standard Schedule", linestyle="--", color="blue")
    ax2.plot(eps_asc_log, label="Adaptive Schedule", color="red", linewidth=2)
    ax2.set_title("Temperature Schedule (Epsilon)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Temperature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# --- 创建 Gradio 界面 ---
with gr.Blocks() as demo:
    gr.Markdown("# 🔥 EPH-ASC: Avoiding Premature Collapse")
    gr.Markdown("Visualizing how Adaptive Stability Control prevents early mode collapse in Sinkhorn layers.")
    
    with gr.Row():
        with gr.Column():
            n_points = gr.Slider(5, 50, value=10, step=1, label="Number of Points (N)")
            decay_rate = gr.Slider(0.8, 0.99, value=0.95, label="Cooling Rate (Alpha)")
            k_safe = gr.Slider(0.1, 2.0, value=0.5, label="Safety Slope (k_safe)")
            btn = gr.Button("Run Simulation", variant="primary")
        
        with gr.Column():
            plot_output = gr.Plot(label="Training Dynamics")
    
    btn.click(simulate, inputs=[n_points, decay_rate, k_safe], outputs=plot_output)

if __name__ == "__main__":
    demo.launch()
# ... (上面是原本的绘图代码) ...

# --- 创建 Gradio 界面 ---
with gr.Blocks() as demo:
    gr.Markdown("# 🔥 EPH-ASC: Avoiding Premature Collapse")
    gr.Markdown("Visualizing how Adaptive Stability Control prevents early mode collapse in Sinkhorn layers.")
    
    with gr.Row():
        with gr.Column():
            n_points = gr.Slider(5, 50, value=10, step=1, label="Number of Points (N)")
            decay_rate = gr.Slider(0.8, 0.99, value=0.95, label="Cooling Rate (Alpha)")
            k_safe = gr.Slider(0.1, 2.0, value=0.5, label="Safety Slope (k_safe)")
            btn = gr.Button("Run Simulation", variant="primary")
        
        with gr.Column():
            plot_output = gr.Plot(label="Training Dynamics")
    
    btn.click(simulate, inputs=[n_points, decay_rate, k_safe], outputs=plot_output)

    # ------------------ 新增：企业服务说明 ------------------
    with gr.Accordion("💼 Want to use this in your Business?", open=False):
        gr.Markdown("""
        ### 🚀 Enterprise Services
        We offer professional support to help you integrate **Adaptive Annealing** into your products.
        
        * **Custom Integration**: Fit EPH-ASC into your model backbone.
        * **Performance Tuning**: Optimized implementations for low-latency environments.
        
        **[Contact Us for Commercial Licensing](mailto:your.email@example.com)**
        """)
    # -------------------------------------------------------

if __name__ == "__main__":
    demo.launch()
