import torch
from torch import nn
import math
from my_code.Softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    # handles keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    # (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    # dimensions (if provided). The implementation should return an output with the shape (batch_size,..., d_v).
    # usually d_k = d_v = d_head = d_model/num_heads


    # Basically does the Softmax( (Q^T @ K) / sqrt(d_k)) * V


    original_dtype = Q.dtype

    # Upcast
    Q32 = Q.to(torch.float32)       # (B, H, T, T)
    K32 = K.to(torch.float32)       # (B, H, T, T)
    V32 = V.to(torch.float32)       # (B, H, T, d_v)


    #last dimention of Q
    d_k = Q.size(-1)


    scores = torch.matmul(Q32, K32.transpose(-1,-2))
    scores = scores / math.sqrt(d_k)                        # (B, H, T, T)

    # boolean masking, if provided
    if mask is not None:
        
        # Accept bool or 0/1 floats (convet to bool if integers)
        if mask.dtype is not torch.bool:
            mask = (mask != 0)

        mask = mask.to(device=scores.device)

        # Prepend singleton dims until mask.ndim == scores.ndim
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        
        
        #neg_inf = torch.finfo(scores.dtype).min                 returns the most negative representable value for that dtype ( supposed to be safer than simply -inf )

        scores = scores.masked_fill(~mask, float('-inf'))             # we want to flip mask to ~mask because we need to fill the -inf at the FALSE locations.       


    # we want a probability distribution over keys for each query
    attn = softmax(scores, dim=-1)

    out32 = torch.matmul(attn, V32)         # (B, H, T, T) @ (B, H, T, d_v) -> (B, H, T, d_v)

    out = out32.to(original_dtype)

    return out

