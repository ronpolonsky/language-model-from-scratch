import argparse
import math
import time
import os
from typing import Optional
import numpy as np
import torch
from array import array

from my_code.transformer_model import TransformerLM
from my_code.crossEntropyLoss import cross_entropy_loss
from my_code.adamw_optimizer import AdamWOptimizer  
from my_code.learning_rate_scheduling import learning_rate_schedule
from my_code.checkpointing import save_checkpoint
from my_code.gradient_clipping import clip_gradients
from my_code.load_batch import get_batch
from my_code.tokenizer import Tokenizer
from scripts.encode_dataset_into_ids import encode_one


def build_parser():

    # Parameters for Data
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type = str, required=True)
    p.add_argument("--val-path", type = str, required=True)
    p.add_argument("--raw_dtype",  type=str, default=None, choices=["uint16", "int32"], help="Only needed for raw .bin token files.")
    p.add_argument("--overwrite", action="store_true", help="Force re-encode .txt even if cached")


    # Model parameters
    p.add_argument("--vocab_size", type = int, required = True)
    p.add_argument("--context_length", type = int, default = 256)
    p.add_argument("--num_layers", type = int, default = 4)
    p.add_argument("--d_model", type = int, default = 512)
    p.add_argument("--num_heads", type = int, default = 16)
    p.add_argument("--d_ff", type = int, default=1344, help="Defaults to 4*d_model if None")
    p.add_argument("--rope_theta", type = float, default = 10000.0)
    p.add_argument("--epsilon", type = float, default = 1e-8)

    # Optimizer + Learning Rate Scheduling parameters
    p.add_argument("--learning_rate", type = float,  default = 1e-3)
    p.add_argument("--betas", type = float, nargs = 2, default = (0.9, 0.95))
    p.add_argument("--optimizer_eps", type = float, default = 1e-8)
    p.add_argument("--weight_decay", type = float, required = True)
    p.add_argument("--lr_min", type = float, default = 0.0)
    p.add_argument("--warmup_steps", required = True, type = int)
    # cosine_steps will equal to  max_steps

    # Train loop
    p.add_argument("--batch_size", type = int, default = 64)
    p.add_argument("--tokens_processed", type = int, default = 327680000)
        #NOTE: here we will use this to set:      max_steps = tokens_processed / (batch_size * context_length)
   
    # Gradient Clipping parameters
    p.add_argument("--max_l2_norm", type = float, required = True)
    p.add_argument("--grad_clip_eps", type = float, default = 1e-6)

    # User provided path for checkpointing
    p.add_argument("--checkpointing_path", type = str, required = True)
    p.add_argument("--save_every", type = int, default=1000)

    # parameters for eval
    p.add_argument("--eval_every", type=int, default=1000, help="How often (steps) to run validation.")
    p.add_argument("--eval_batches", type=int, default=50, help="How many batches to average for validation loss.")

    # Tokenizer (needed to get ids from dataset_path)
    p.add_argument("--vocab", type=str, help="Path to vocab.tsv (required if train/val are .txt).")
    p.add_argument("--merges", type=str, help="Path to merges.txt (required if train/val are .txt).")
    p.add_argument("--special", type=str, default="<|endoftext|>", help="Special token string.")
    p.add_argument("--cache_dir", type=str, default="data/encoded", help="Where to write encoded token id .bin files if inputs are .txt")

    # System
    p.add_argument("--device", type=str, default="cuda:0")
    

    return p



def load_or_encode_ids(path: str,
                       tok: Tokenizer | None = None,
                       raw_dtype: str | None = None,
                       cache_dir: str = "data/encoded",
                       mmap: bool = True,
                       overwrite: bool = False) -> np.ndarray:
    if path.endswith(".npy"):
        print(f"[data] loading numpy ids: {path}", flush=True)
        return np.load(path, mmap_mode="r" if mmap else None)

    if path.endswith(".bin"):
        if raw_dtype is None:
            raise ValueError("For .bin, specify raw_dtype as 'uint16' or 'int32'.")
        np_dtype = {"uint16": np.uint16, "int32": np.int32}[raw_dtype]
        n = os.path.getsize(path) // np.dtype(np_dtype).itemsize
        print(f"[data] memmap raw bin: {path}  dtype={np_dtype}  n={n}", flush=True)
        return np.memmap(path, dtype=np_dtype, mode="r", shape=(n,))

    if path.endswith(".txt"):
        if tok is None:
            raise ValueError("Encoding .txt requires a Tokenizer (provide --vocab/--merges).")
        # Use your streaming encoder with progress logging
        out_bin_path = encode_one(path, cache_dir, tok, overwrite=overwrite)  # returns Path
        out_bin_path = str(out_bin_path)
        n = os.path.getsize(out_bin_path) // np.dtype(np.uint16).itemsize
        return np.memmap(out_bin_path, dtype=np.uint16, mode="r", shape=(n,))

    raise ValueError(f"Unsupported input: {path}. Use .txt, .npy, or .bin.")


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, val_data: np.ndarray, batch_size: int, context_length: int, device: str, K: int) -> float:
    """
        validation loop (averages loss over K batches)
    """
    model.eval()
    losses = []
    for _ in range(K):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)





def main(args):
    """ Performs One Loop:
        one forward + loss + backward + one optimizer.step() on ONE batch.
        (One epoch is doing these steps until we covered whole dataset with our bathces-> usually ~ N/B times)
        """

    device = args.device

    # build tokenizer only if any input is .txt (otherwise assume content is already token_ids and we do not need to encode them)
    need_tok = args.train_path.endswith(".txt") or args.val_path.endswith(".txt")
    tok = None
    if need_tok:
        if not args.vocab or not args.merges:
            raise ValueError("Provide --vocab and --merges when using .txt inputs.")
        tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[args.special])

    train_data = load_or_encode_ids(args.train_path, tok, args.raw_dtype, cache_dir=args.cache_dir, mmap=True, overwrite=args.overwrite)
    val_data   = load_or_encode_ids(args.val_path,   tok, args.raw_dtype, cache_dir=args.cache_dir, mmap=True, overwrite=args.overwrite)


    
    if args.d_ff is not None:
        d_ff = args.d_ff  
    else:
        d_ff = 4 * args.d_model


    transformer_model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        epsilon=args.epsilon,
        device=device,
        dtype=torch.float32,
    )
    transformer_model.to(device)

    params = transformer_model.parameters()

    #if using eval() later on
    transformer_model.train()

    optimizer = AdamWOptimizer(params, lr=args.learning_rate, betas=tuple(args.betas), eps=args.optimizer_eps, weight_decay=args.weight_decay)



    os.makedirs(args.checkpointing_path, exist_ok=True)  # ensure directory exists

    max_steps = args.tokens_processed // (args.batch_size * args.context_length)
    Tc = int(max_steps) - 1  # cosine ends on the last training step


    for iteration in range(max_steps):
        
        # get one batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # set learning rate for this step in each parameter's state
        # Learning rate schedule
        learning_rate = learning_rate_schedule(iteration, args.learning_rate, args.lr_min, args.warmup_steps, Tc)  

        # set leraning rate so it is used in the step()
        optimizer.set_lr(learning_rate)

        # run one forward pass
        logits = transformer_model(x)

        # get loss of this currennt epoch (loop)
        loss = cross_entropy_loss(logits, y)

        # clear old grads (PyTorch accumulates by default from previous loop)
        optimizer.zero_grad(set_to_none=True)

        # compute gradients with forwards pass and resulting logits
        loss.backward()   # I did not implement this, but this is inherited from pytorch and my usage of autograd


        #clip gradients
        if args.max_l2_norm and args.max_l2_norm > 0:
            clip_gradients(transformer_model.parameters(), args.max_l2_norm, args.grad_clip_eps)
        
        # optimizer step  (with already stored new learning rate in parameters states)
        optimizer.step()

        # save checkpoint
        if (iteration + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpointing_path, f"ckpt_step_{iteration+1:07d}.pt")
            save_checkpoint(transformer_model, optimizer, iteration=iteration+1, out=ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}")
        
        # some logging to see progress 
        if (iteration % 100) == 0: 
            print(f"[train] step {iteration+1:7d}/{int(max_steps)}  loss {loss.item():.4f}  lr {learning_rate:.2e}", flush=True)

        # Eval (every eval_every)
        if (iteration + 1) % args.eval_every == 0:
            val_loss = evaluate_loss(transformer_model, val_data, args.batch_size, args.context_length, device, args.eval_batches)
            print(f"[val] step {iteration+1:>7d}  val_loss {val_loss:.4f}")

    # Final checkpoint
    final_ckpt = os.path.join(args.checkpointing_path, f"ckpt_final_step_{int(max_steps):07d}.pt")
    save_checkpoint(transformer_model, optimizer, iteration=int(max_steps), out=final_ckpt)
    print(f"done. saved final checkpoint to {final_ckpt}")







if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args() # tells argparse to read the command line, 
                               # match it against all the arguments we declared with add_argument(...), 
                               # and return the results as a Namespace (an object with attributes).
    main(args)