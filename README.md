# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~40M tokens processed.

I wrote the byte-level BPE tokenizer, AdamW optimizer, and Transformer (Multi-Head Self-Attention, Pre-Norm blocks, RoPE) without PyTorch convenience modules—i.e., no nn.MultiheadAttention, no torch.optim.AdamW, no schedulers, no external tokenizers—using only basic tensor ops and nn.Parameter.

Both the tokenizer and the model are trained on TinyStories
