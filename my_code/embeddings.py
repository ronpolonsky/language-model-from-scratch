import torch
from torch import nn


class Embeddings(nn.Module):

    # num_embeddings: int Size of the vocabulary
    # embedding_dim (vocab_size): int Dimension of the embedding vectors, i.e., dmodel
    # device: torch.device | None = None Device to store the parameters on
    # dtype: torch.dtype | None = None Data type of the parameters


    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be > 0, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")

        super().__init__()

        init_mean = 0          # both given by handout
        init_std = 1


        # ? Note: Need to always have dtype as fp32 ? 

        weights = torch.empty(num_embeddings, embedding_dim, device = device, dtype=torch.float32)

        nn.init.trunc_normal_(weights, mean=init_mean, std=init_std)

        self.Weights = nn.Parameter(weights)




    def forward(self, x):
        # input is a grid of integer IDs of shape (batch_size, sequence_length).
        # returns a grid of vectors, shape (batch_size, sequence_length, d_model).

        out = self.Weights[x]
        return out