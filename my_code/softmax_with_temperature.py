import torch
import math



def softmax_with_temp(last_logits, tau):
    """
    Temperature-scaled softmax along the last dimension.

    Args:
        logits: Tensor of shape (..., V) — last dim is the vocabulary/classes.
        tau:    Temperature > 0. Smaller -> sharper (more greedy), larger -> flatter.

    Returns:
        probs:  Tensor of shape (..., V), same dtype as `logits`, rows sum to 1.
    """
    if tau <= 0.0:
        raise ValueError(f"Temperature must be > 0, got {tau}")


    # Work in fp32 for stability, but return in the original dtype
    out_dtype = last_logits.dtype
    x = last_logits.to(dtype=torch.float32)

    x_max = torch.amax(x, dim=-1, keepdim=True)          # (..., 1)
    x_shifted = x - x_max                                 # (..., V)
    x_scaled = x_shifted / tau

    exp_x = torch.exp(x_scaled)

    norm_factor = torch.sum(exp_x, dim=-1, keepdim=True).clamp_min(1e-12)

    probs = exp_x / norm_factor

    return probs.to(dtype=out_dtype)
