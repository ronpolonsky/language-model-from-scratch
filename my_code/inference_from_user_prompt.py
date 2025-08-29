import argparse
import torch
import sys

from my_code.tokenizer import Tokenizer
from my_code.transformer_model import TransformerLM
from my_code.generate_tokens import generate_tkns
from my_code.checkpointing import load_checkpoint


CKPT_PATH   = "saved_checkpoints/ckpt_final_step_0005000.pt"
VOCAB_PATH  = "trained_tokenizer/vocab.tsv"
MERGES_PATH = "trained_tokenizer/merges.txt"

def inference_from_input(user_text: str) -> str:
    # same defaults as before
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

    device = "cpu"
    transformer_model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        epsilon=1e-8,
        device=device,
        dtype=torch.float32
    ).to(device)

    _ = load_checkpoint(CKPT_PATH, transformer_model, load_optimizer=False, optimizer=None)

    # Encode input
    prompt_ids = tokenizer.encode(user_text)
    if len(prompt_ids) >= 256:
        prompt_ids = prompt_ids[-255:]

    prompt_ids_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    try:
        eos_id = tokenizer.bytes_to_id["<|endoftext|>".encode("utf-8")]
    except KeyError:
        eos_id = -1

    with torch.no_grad():
        generated_ids = generate_tkns(transformer_model, prompt_ids_tensor, eos_id,
                                      max_new_tokens=250, temperature=0.8, top_p=0.95,
                                      device=device)

    return tokenizer.decode(generated_ids)
