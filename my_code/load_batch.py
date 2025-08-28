import numpy as np
import torch

def get_batch(dataset,  batch_size, context_length, device):
    """
    Sample ONE batch of next-token prediction pairs from a long token stream.

    Returns:
        inputs  : LongTensor of shape (batch_size, context_length)
        targets : LongTensor of shape (batch_size, context_length)
    """
    if dataset.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {dataset.shape}")
    if context_length <= 0 or batch_size <= 0:
        raise ValueError("context_length and batch_size must be positive")
        
    n = dataset.shape[0]

    if n < context_length + 1:
        raise ValueError(
            f"sequence too short: need at least context_length+1={context_length+1} tokens, got {n}"
        )


    B = batch_size
    T = context_length

    batch_seqs = np.empty((B,T), dtype = dataset.dtype)
    batch_targets = np.empty((B,T), dtype = dataset.dtype)

    for i in range(B):
        start_indx = np.random.randint(0, n - T)
        seq = dataset[start_indx: start_indx + T]
        trgts = dataset[start_indx + 1 : start_indx + 1 + T]
        
        batch_seqs[i,:] = seq
        batch_targets[i, :] = trgts


    if batch_seqs.dtype != np.int64:
        batch_seqs = batch_seqs.astype(np.int64, copy=False)
        batch_targets = batch_targets.astype(np.int64, copy=False)
    
    seqs = torch.from_numpy(batch_seqs).to(device=device, dtype = torch.long)
    targets = torch.from_numpy(batch_targets).to(device=device, dtype = torch.long)
    return seqs, targets
