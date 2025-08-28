import torch 
from torch import nn

from my_code.embeddings import Embeddings
from my_code.linear import Linear
from my_code.RoPE import RotaryPositionalEmbedding
from my_code.RMSnorm import RMSNorm
from my_code.Softmax import softmax
from my_code.transformer_block import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, epsilon = 1e-5, device = None, dtype = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.epsilon = epsilon
        
        d_head = self.d_model // self.num_heads

        self.token_embeddings = Embeddings(self.vocab_size, self.d_model, device, dtype)
        self.rope = RotaryPositionalEmbedding(self.theta, d_head, self.sequence_length, device)

        blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.rope, self.epsilon, device, dtype)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(self.d_model, epsilon = epsilon, device= device, dtype = dtype)
        
        self.lm_head = Linear(self.d_model, self.vocab_size, device, dtype)

    def forward(self, token_ids):

        device = token_ids.device

        B, T= token_ids.shape

        x = self.token_embeddings(token_ids)
        token_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        
        for block in self.blocks:
            x = block(x, token_positions)

        # after attention blocks: RMSNorm + LM head + Softmax
        y = self.final_norm(x)
        logits = self.lm_head(y)

        # note that it returns logits. Not the final probabilities after softmax
        return logits
        