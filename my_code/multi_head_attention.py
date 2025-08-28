import torch
from torch import nn
from einops import einsum, rearrange
from my_code.linear import Linear
from my_code.RoPE import RotaryPositionalEmbedding
from my_code.Scaled_Dot_Product_Attention import scaled_dot_product_attention




class MultiHeadAttention(nn.Module):


    def __init__(self, d_model, num_heads, rope: RotaryPositionalEmbedding, device=None, dtype = None):

        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        # d_model = dimensionality of the transformer block
        # input is (B,T, D)
        # d_head = d_model / num_heads
        # each Q, K, V in attention head is (B, T, d_head)
        # num heads is the num of heads run in parrallel where each head has its own Q,K,V (which we stack in one mbig matrix).

       
        # Note: remember that the class Linear cares only about the last dimension of the input and last dimension of the output!
        # This is because teh matrix we multiply with is dimensino (d_model, d_model)
        self.Q_proj = Linear(d_model, d_model, device = device, dtype = dtype)
        self.K_proj= Linear(d_model, d_model, device = device, dtype = dtype)
        self.V_proj = Linear(d_model, d_model, device = device, dtype = dtype)

        # last output projection that projects the concatenation 
        self.O_proj = Linear(d_model, d_model, device = device, dtype = dtype)

        self.n_heads = num_heads
        self.d_model = d_model
        self.rope = rope


    def forward(self, x, token_positions):
        
        # x = (B, T, d_model)
        B, T, _ = x.shape
        
        h = self.n_heads
        # each self.Q.W is (H * d_head, d_model)
        # thus we need x @ Q.W^T ---->  (B, T, d_model) * (d_model, H*d_head) = (B, T, H*d_head)
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)


        #  Split heads and move head to its own axis
        #  (B, T, h*d_head) -> (B, h, T, d_head)
        Q = rearrange(Q, 'b t (h k) -> b h t k', h=h)
        K = rearrange(K, 'b t (h k) -> b h t k', h=h)
        V = rearrange(V, 'b t (h k) -> b h t k', h=h)


        # Apply RoPE to Q and K

        # Note: Add checks to token_positions?
        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)



        # Build causal Masking 
        # We can create a lower triangular with True of shape (seq_length, seq_length) and it will boradcast this to (....., seq_len, se    _len).
        all_ones = torch.ones(T, T, device = x.device, dtype = x.dtype)

        # set all values in diagonal and under to 0.
        mask = torch.triu(all_ones, diagonal = 1)

        mask = ~mask.to(torch.bool)  # need to invert

        attentions = scaled_dot_product_attention(Q, K, V, mask)   #(B, h, T, d_v)

        concat_attn = rearrange(attentions, 'b h t d -> b t (h d)') #(B, h, T, d_model)

        y = self.O_proj(concat_attn)

        return y