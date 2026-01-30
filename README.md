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
