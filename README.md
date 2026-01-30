# torch-sinkhorn-asc

**Avoiding Premature Collapse: Adaptive Annealing for Entropy-Regularized Structural Inference**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a PyTorch implementation of **Efficient PH-ASC (Piecewise Hybrid Adaptive Stability Control)**.

It addresses the "Premature Mode Collapse" problem in differentiable matching layers (Sinkhorn) by dynamically monitoring the thermodynamic speed limit of the inference process. By enforcing a linear stability law, it prevents the inference trajectory from locking into spurious local basins during annealing.

## Features

* **Log-Domain Sinkhorn**: Numerically stable implementation for entropy-regularized optimal transport.
* **EPH-ASC Scheduler**: An adaptive controller that monitors distributional drift.
* **Thermodynamic Braking**: Automatically detects instability ($||\Delta_t|| > k_{safe} \cdot \epsilon$) and pauses cooling to recover the signal.
* **Efficient**: Decouples expensive spectral diagnostics from the training loop, reducing overhead from $O(N^3)$ to amortized $O(1)$.

## Installation

```bash
git clone [https://github.com/xxx0438/torch-sinkhorn-asc.git](https://github.com/xxx0438/torch-sinkhorn-asc.git)
cd torch-sinkhorn-asc
pip install -r requirements.txt

Directory Structure
torch-sinkhorn-asc/
├── src/
│   ├── sinkhorn.py      # Log-domain Sinkhorn solver
│   └── scheduler.py     # EPH-ASC Adaptive Scheduler
├── examples/
│   └── train_demo.py    # Training loop simulation
├── requirements.txt
└── setup.py

Usage
import torch
from src.sinkhorn import SinkhornLayer
from src.scheduler import EPH_ASC_Scheduler

# 1. Initialize layer and scheduler
# k_safe=0.5 is the recommended default from the paper [cite: 137]
sinkhorn = SinkhornLayer(n_iters=20)
scheduler = EPH_ASC_Scheduler(
    init_epsilon=1.0,
    min_epsilon=0.01,
    decay_rate=0.95,
    k_safe=0.5
)

# 2. Mock training loop
for epoch in range(100):
    # Get current temperature
    curr_eps = scheduler.epsilon

    # Forward pass
    # cost_matrix shape: (Batch, N, N)
    P = sinkhorn(cost_matrix, curr_eps)

    # ... Compute Loss and Optimizer Step ...

    # 3. Update Scheduler (Check drift and adjust temperature)
    # Drift is monitored as ||P_t - P_{t-1}||_F (see Eq. 2 in paper [cite: 121])
    new_eps = scheduler.step(P)

Running the Demo
python examples/train_demo.py

Reference
This code implements the algorithms described in:


Avoiding Premature Collapse: Adaptive Annealing for Entropy-Regularized Structural Inference  Abstract: Differentiable matching layers, often implemented via entropy-regularized Optimal Transport, serve as a critical approximate inference mechanism in structural prediction. However, recovering discrete permutations via annealing is notoriously unstable... We identify a fundamental mechanism for this failure: Premature Mode Collapse.

## 🤝 Enterprise Services & Commercial Support

While this open-source library provides a research-grade implementation of EPH-ASC [1], we offer dedicated services for enterprise clients who require production-ready solutions.

**Our Services Include:**
* **Production Optimization**: Custom CUDA kernel implementations for extreme performance (beyond standard PyTorch).
* **Integration Support**: Assistance with integrating Sinkhorn layers into your existing pipelines (Visual Matching, Permutation Learning, Ranking).
* **Custom Stability Control**: Tuning the $k_{safe}$ parameter and safety schedules for your specific datasets.
* **Consulting**: Expert guidance on avoiding "Mode Collapse" in large-scale structural inference tasks.

**Contact Us:**
📧 Email: [your.email@example.com](liuyizhi774@gmail.com)

License

This project is released under a **dual-licensing model**:

### 1. Open Source License (AGPL-3.0)

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

You are free to use, modify, and distribute this software under the terms of the AGPL-3.0.
If you use this software to provide a network service (e.g. SaaS, API, or online demo),
you must make the complete corresponding source code available under the same license.

See the `LICENSE-AGPL` file for details.

### 2. Commercial License

For commercial use cases that do **not** comply with the AGPL-3.0 (e.g. closed-source use,
proprietary products, internal enterprise deployment, or SaaS without source disclosure),
a separate **commercial license** is required.

Please contact the author to obtain a commercial license.OVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
