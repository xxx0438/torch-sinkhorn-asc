import torch
import torch.nn as nn

def log_sinkhorn(log_alpha, n_iters=20):
    """
    Log-space Sinkhorn iteration for numerical stability.
    Standard Sinkhorn suffers from underflow when epsilon -> 0.
    """
    for _ in range(n_iters):
        # Row normalization (in log space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization (in log space)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()

class SinkhornLayer(nn.Module):
    def __init__(self, n_iters=20):
        super().__init__()
        self.n_iters = n_iters

    def forward(self, cost_matrix, epsilon):
        """
        Args:
            cost_matrix: (B, N, N) input cost.
            epsilon: Temperature parameter.
        """
        # Formulation: P = exp(-C / epsilon)
        # We work in log space: log_P = -C / epsilon
        log_alpha = -cost_matrix / epsilon
        
        P = log_sinkhorn(log_alpha, self.n_iters)
        return P
