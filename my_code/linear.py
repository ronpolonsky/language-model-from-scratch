import math
import torch
from torch import nn

__all__ = ["Linear"]

#linear layer: y = Wx ---> but does y = x@W^T    

class Linear(nn.Module):
    # my object now 'extends' nn.Module,
    # so my class inherits all of nn.Modul's methods and behaviors.
        


        def __init__(self, dim_in, dim_out, device=None, dtype=None):
            
            # in_features: int final dimension of the input
            # out_features: int final dimension of the output
            # device: torch.device | None = None Device to store the parameters on
            # dtype: torch.dtype | None = None Data type of the parameters



            super().__init__()
            # This is like basically calling the parent's (nn.Module's) constructor
            # it is crucial to sets up the hidden machinery that nn.Modul carefuly 
            # takes care of.
            

            
            # Create the learnable weight matrix W with the required shape.
            # torch.empty allocates uninitialized memory - which we fill it below.
            # Storing as (out_features, in_features) matches the W, not W^T as in handout.
            self.W = nn.Parameter(torch.empty(dim_out, dim_in, device = device, dtype = dtype))



            # according to handout:  std = sqrt( 2 / (d_in + d_out) )
            std = math.sqrt( 2 / (dim_in + dim_out) )

            # truncate to [-3*std, +3*std] around mean 0.
            nn.init.trunc_normal_(self.W, mean = 0.0, std=std, a = -3*std, b = 3*std)


        def forward(self, x: torch.Tensor): 
            #It's equivalent to doing a batched matmul  x @ W^T  that maps 
            #(B,T,in_features) to (B,T,out_features),
             


            # einsum expresses the batched matmul without manual reshapes.
            #    Pattern:  "... i, o i -> ... o"
            #      - "... i"    : input x, last dim 'i' = in_features
            #      - "o i"      : weight W, with dims (out_features='o', in_features='i')
            #      - "... o"    : output keeps all leading dims, last dim becomes 'o'
            #
            #    This is equivalent to:
            #      for 2D x: x @ W.T
            #      for ND x: torch.matmul(x, W.T) after appropriate reshaping
            #    but einsum keeps it clean and general.

            return torch.einsum('... i, o i -> ... o', x, self.W)

