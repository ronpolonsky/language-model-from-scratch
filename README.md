# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~20M Parameters and ~40M tokens processed.

I wrote the byte-level BPE tokenizer, AdamW optimizer, and Transformer with Multi-Head Self-Attention, Pre-Norm blocks, and RoPE without PyTorch's module definitions (no nn.MultiheadAttention, no torch.optim.AdamW, no schedulers, no external tokenizers) - only using basic tensor ops and nn.Parameter inheritance from nn.Module

Both the tokenizer and the model are trained on TinyStories downloaded from Hugging Face: https://huggingface.co/datasets/roneneldan/TinyStories


##To Run the Demo

Clone the repo and start the interactive app:

```bash
git clone https://github.com/ronpolonsky/language-model-from-scratch.git
cd language-model-from-scratch
git lfs install
git lfs pull
pip install -r requirements.txt
streamlit run myApp.py

