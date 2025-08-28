import torch
from torch import nn
from my_code.RMSnorm import RMSNorm
from my_code.multi_head_attention import MultiHeadAttention
from my_code.ffn import Swiglu


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, rope=None,  epsilon = 1e-5, device = None, dtype = None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.norm_attn = RMSNorm(d_model, epsilon = epsilon, device= device, dtype = dtype)
        self.multi_attn = MultiHeadAttention(d_model, num_heads, rope = rope, device= device, dtype = dtype)

        self.norm_ffn = RMSNorm(d_model, epsilon = epsilon, device= device, dtype = dtype)
        self.ffn = Swiglu(d_model, d_ff, device=device, dtype = dtype)


    def forward(self, x, token_positions):

        x_norm = self.norm_attn(x)

        # add residual of multi head attention
        post_attn_x = self.multi_attn(x_norm, token_positions)

        x = x + post_attn_x

        x_norm_ffn = self.norm_ffn(x)

        # add residual of FFN
        y  = x + self.ffn(x_norm_ffn)

        return y