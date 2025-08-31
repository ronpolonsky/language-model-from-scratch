# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~20M Parameters and ~40M tokens processed.

I implemented the byte-level BPE tokenizer, AdamW optimizer, and Transformer with Multi-Head Self-Attention, Pre-Norm blocks, and RoPE - without using PyTorch's built in modules (no nn.MultiheadAttention, no torch.optim.AdamW, no schedulers, no external tokenizers) - only using basic tensor operatins and inheritance from nn.Module.

Both the tokenizer and the model are trained on TinyStories downloaded from Hugging Face: https://huggingface.co/datasets/roneneldan/TinyStories


##To Run the Demo

Clone the repo and start the interactive app (_requires git lfs_)

```bash
git clone https://github.com/ronpolonsky/language-model-from-scratch.git
cd language-model-from-scratch
git lfs install
git lfs pull
pip install -r requirements.txt
streamlit run myApp.py

