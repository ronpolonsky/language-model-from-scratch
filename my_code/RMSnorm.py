import torch
from torch import nn


class RMSNorm(nn.Module):
    # RMS layer normalization
    # Normalize only by RMS (without mean subtracdtion)
    # apply learnable D dimesional vector g_i

    def __init__(self, d_model: int , epsilon: float = 1e-5, device=None, dtype=None):

        # Args:
        #   d_model: hidden size (D) last dimension of the inputs we normalize
        #   eps: small constant to avoid division by zero
        #   device/dtype: where/how to store the parameter (optional)
        
        super().__init__()
        self.d_model = d_model
        self.eps = epsilon

        # Learnable per-feature scale (gain). Shape: (D,)
        # Handout says RMSNorm init = 1, so we start with all ones.
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    
    def forward(self, x: torch.Tensor):
        # Args:
        #    x: Tensor of shape (..., D). In the Transformer it is (B, T, D).
        # Returns:
        #   Tensor of same shape and dtype as x.
    

        #remmeber the callers type
        original_dtype = x.dtype


        # As per handout -> upcast to float32 for stable squares 
        x_fp32 = x.to(torch.float32)


        # Now compute RMS
        
        # 1) compute squares element wise
        squared = x_fp32 * x_fp32

        # 2) average across the last dimension (feature dimension) 
        mean_sq = squared.mean(dim = -1, keepdim=True)

        # 3) add epsilon 
        mean_sq_eps = mean_sq + self.eps

        rms = torch.sqrt(mean_sq_eps)


        # 4) Normalize 
        normalized_x = x_fp32/rms  # (..., D) / (..., 1) -> (..., D) (broadcasted)

        # 5) Apply learnable gain per feature. self.weight has shape (D,)
        # #    Broadcasting multiplies the last dimension by weight for every (B, T).
        g = self.gain.to(x_fp32.dtype)
        out = normalized_x * g


        # 6) Cast back to the original dtype so the rest of the model stays consistent
        return out.to(original_dtype)



