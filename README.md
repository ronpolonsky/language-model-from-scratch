# language-model-from-scratch
This repo implements a small GPT-style language model trained on TinyStories (Eldan & Li, 2023), end-to-end from scratch, with ~20M Parameters and ~40M tokens processed.

I implemented the byte-level BPE tokenizer, AdamW optimizer, and the Transformer with Multi-Head Self-Attention, Pre-Norm blocks, RoPE and Scheduler without using PyTorch's built in modules (no nn.MultiheadAttention, no torch.optim.AdamW, no external tokenizers) - only using basic tensor operatins and inheritance from nn.Module.

Both the tokenizer and the model are trained on TinyStories downloaded from Hugging Face: https://huggingface.co/datasets/roneneldan/TinyStories

## To Run the Demo:

**In Google Colab with UI cell** (no instls)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/ronpolonsky/language-model-from-scratch/blob/main/interactive_app/myApp_colab.ipynb)

---

**Run Locally** _(Use python3 on mac/linux or py on Windows if needed)_ ---> takes time to install lfs..

```bash
git clone https://github.com/ronpolonsky/language-model-from-scratch.git
cd language-model-from-scratch
git lfs install
git lfs pull
pip install -r requirements.txt
streamlit run myApp.py
