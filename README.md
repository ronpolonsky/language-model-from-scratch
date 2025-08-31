# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~20M Parameters and ~40M tokens processed.

I implemented the byte-level BPE tokenizer, AdamW optimizer, and the Transformer with Multi-Head Self-Attention, Pre-Norm blocks, RoPE and Scheduler without using PyTorch's built in modules (no nn.MultiheadAttention, no torch.optim.AdamW, no schedulers, no external tokenizers) - only using basic tensor operatins and inheritance from nn.Module.

Both the tokenizer and the model are trained on TinyStories downloaded from Hugging Face: https://huggingface.co/datasets/roneneldan/TinyStories

## To Run the Demo:

**in Google Colab (no installs)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/ronpolonsky/language-model-from-scratch/blob/main/interactive_app/myApp.ipynb)

---

**Run Locally**

```bash
git clone https://github.com/ronpolonsky/language-model-from-scratch.git
cd language-model-from-scratch
git lfs install && git lfs pull
python -m ensurepip --upgrade
python -m pip install -r interactive_app/requirements.txt
python -m streamlit run interactive_app/myApp.py

