import torch 


def softmax(x: torch.Tensor, dim):
    """
    Numerically-stable softmax along a given dimension.
    - Works for any shape
    - Keeps the same shape as input
    - No torch.nn.functional.softmax; implemented from primitives
    """

    original_type = x.dtype
    x32 = x.to(torch.float32)

    # keepdim=True makes the reduced dimension stick around with size 1, whcih is good for broadcasting in the next line
    x_max, _ = torch.max(x32, dim=dim, keepdim= True)

    x_stable = x - x_max

    exps = torch.exp(x_stable)

    # Sum of exponentials along 'dim' (denominator of softmax).
    denominators = torch.sum(exps, dim=dim, keepdim=True)

    #   (Optional, defensive) clamp denom to avoid divide-by-zero if a slice is all -inf
    #    (this happens in pathological masking cases; usually your mask guarantees at least one True).
    # tiny = torch.finfo(exps.dtype).tiny
    #denominators = torch.clamp(denom, min=tiny)

    results32 = exps / denominators

    return results32.to(original_type)

