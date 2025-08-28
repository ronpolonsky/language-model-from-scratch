import torch
from torch import nn
from my_code.linear import Linear  


class Swiglu(nn.Module):
    
    # FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )
    # x: (..., d_model) -> (..., d_model

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        # Reuse my Linear implementatoin (matrix mult).
        self.w1 = Linear(d_model, d_ff, device=device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype = dtype)

    # Note: NEED TO SET D_FFF = 8/3 * D_MODEL?? OR CAN ASSUME IT IS PASSED TO US?

    def forward(self, x):

        a = self.w1(x)
        gate = a*torch.sigmoid(a)
        z = gate * self.w3(x)

        y = self.w2(z)

        return y
