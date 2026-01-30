import torch

class EPH_ASC_Scheduler:
    """
    Efficient Piecewise Hybrid - Adaptive Stability Control (EPH-ASC).
    
    Implements the linear stability law described in the paper:
    ||Delta_t||_F <= k_safe * epsilon_t
    """
    def __init__(self, init_epsilon, min_epsilon, decay_rate, k_safe=0.5):
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.k_safe = k_safe  # Dataset-specific safety slope (default 0.5)
        
        self.prev_P = None
        self.history = {"epsilon": [], "drift": [], "braking": []}

    def step(self, current_P):
        """
        Update temperature based on distributional drift.
        Args:
            current_P: (B, N, N) Transport plan from current step.
        Returns:
            new_epsilon: Updated temperature.
        """
        if self.prev_P is None:
            self.prev_P = current_P.detach()
            return self.epsilon

        # 1. Calculate Drift ||Delta_t|| (Frobenius norm)
        # "The distance between consecutive optimal plans" (Section 3.1)
        drift = torch.norm(current_P.detach() - self.prev_P, p='fro').item()
        
        # 2. Stability Threshold: tau = k_safe * epsilon
        threshold = self.k_safe * self.epsilon
        
        # 3. Check Condition (Section 4.2)
        if drift > threshold:
            # UNSTABLE: Trigger "Thermodynamic Pause" (Braking)
            # Hold epsilon constant to allow signal to mature
            is_braking = True
        else:
            # STABLE: Proceed with standard exponential cooling
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            is_braking = False
            
        # Update state
        self.prev_P = current_P.detach()
        self.history["epsilon"].append(self.epsilon)
        self.history["drift"].append(drift)
        self.history["braking"].append(is_braking)
        
        return self.epsilon
