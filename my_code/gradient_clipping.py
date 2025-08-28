import torch


torch.no_grad()
def clip_gradients(parameters, max_l2_norm, eps: float = 1e-6):
    """
    Global L2-norm gradient clipping (in place).

    Args:
        parameters : iterable of nn.Parameter (e.g., model.parameters())
        max_norm   : M, the maximum allowed global L2 norm for all grads combined
        eps        : numeric stability (default 1e-6, as requested)


    If norm greater than max_norm, rescales all gradiens by (max_norm / (||g||_2 + eps)) --> In-Place.
    """

    # collect all gradients
    grads = [p.grad for p in parameters if (p is not None and p.grad is not None)]

    if not grads:
        raise ValueError(f"no gradients")
    

    device = grads[0].device
    total_sum = torch.zeros((), device=device, dtype=torch.float32)


    for g in grads:
        if g.is_sparse:
            raise RuntimeError("Sparse gradients not supported in this clipper.")

        # detach so that it does not track the math in autograd
        # raise to power of 2 and sum all elements
        total_sum.add_(torch.sum((g.detach().to(dtype=torch.float32)) ** 2))
    
    total_norm = torch.sqrt(total_sum)
    
    if total_norm > max_l2_norm:
        scale = (max_l2_norm / (total_norm + eps))
        for g in grads:
            g.mul_(scale)

    


