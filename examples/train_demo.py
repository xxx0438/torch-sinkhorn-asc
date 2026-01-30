import torch
import sys
import os

# Allow importing from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from sinkhorn import SinkhornLayer
from scheduler import EPH_ASC_Scheduler

def run_simulation():
    print("Initializing EPH-ASC Simulation...")
    
    # Setup consistent with paper experiments (Section 5.1)
    batch_size = 4
    n_points = 10
    n_epochs = 50
    
    # Initialize components
    sinkhorn = SinkhornLayer(n_iters=20)
    scheduler = EPH_ASC_Scheduler(
        init_epsilon=1.0, 
        min_epsilon=0.01, 
        decay_rate=0.95, # Standard exponential schedule
        k_safe=0.5       # Safety slope
    )
    
    for epoch in range(n_epochs):
        # 1. Simulate a Cost Matrix (e.g., from a neural network)
        # Noise reduces over time to simulate learning
        noise_level = max(0.1, 1.0 - epoch / 20.0)
        cost_matrix = torch.randn(batch_size, n_points, n_points) * noise_level
        
        # 2. Get current temperature
        current_eps = scheduler.epsilon
        
        # 3. Forward Pass
        P = sinkhorn(cost_matrix, current_eps)
        
        # 4. Adaptive Update
        # The scheduler decides whether to cool down or brake
        new_eps = scheduler.step(P)
        
        # Status log
        status = "🛑 PAUSED (Braking)" if new_eps == current_eps and current_eps > 0.01 else "❄️ COOLING"
        print(f"Epoch {epoch+1:02d}: eps={current_eps:.4f} | Drift check: {status}")

if __name__ == "__main__":
    run_simulation()
