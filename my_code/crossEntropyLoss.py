import torch

    #Cross-entropy (per token) from logits (no explicit softmax):
    # Given logits O in R^V and true target index t:
    #   loss = -log(softmax(o)[t]) = log( sum {exp(o[a])} ) - o[t]

def cross_entropy_loss(logits, targets):
    # logits of all targets (in matrix form)

    logits = logits.to(dtype=torch.float32)

    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)      # (..., 1)
    shifted_logits = logits - max_logits

    sum_exps = sum_exp = torch.exp(shifted_logits).sum(dim=-1)
    log_normalized = torch.log(sum_exps)

    # targets is (B, T) -> one class id per position. 
    # To use it as an index along the last dimension, we add a trailing dimension: (B, T, 1)
    targets = targets.unsqueeze(-1)
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets) # Note that this returns (... ,1) so need to remove last dimension
    target_logits = target_logits.squeeze(-1)


    per_position_loss = log_normalized - target_logits
    batch_avg_loss = torch.mean(per_position_loss)
    return batch_avg_loss