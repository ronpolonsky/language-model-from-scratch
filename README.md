# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~40M tokens processed.

I wrote the byte-level BPE tokenizer, AdamW optimizer, and Transformer with Multi-Head Self-Attention, Pre-Norm blocks, and RoPE without PyTorch's module definitions (no nn.MultiheadAttention, no torch.optim.AdamW, no schedulers, no external tokenizers) - only using basic tensor ops and nn.Parameter.

Both the tokenizer and the model are trained on TinyStories
