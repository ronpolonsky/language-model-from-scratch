import argparse
import torch
import sys

from my_code.tokenizer import Tokenizer
from my_code.transformer_model import TransformerLM
from my_code.generate_tokens import generate_tkns
from my_code.checkpointing import load_checkpoint


CKPT_PATH   = "saved_checkpoints/tinystories_cpu/ckpt_final_step_0005000.pt"
VOCAB_PATH  = "outputs/tokenizers/tinystories-10k/vocab.tsv"
MERGES_PATH = "outputs/tokenizers/tinystories-10k/merges.txt"


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--user_text", type = str, required = True)


    # Tokenizer parameters
    parser.add_argument("--vocab_path", default = VOCAB_PATH)
    parser.add_argument("--merges_path", default = MERGES_PATH)
    parser.add_argument("--special_tokens", type = str, default = "<|endoftext|>")

    parser.add_argument("--checkpoint_path", default = CKPT_PATH)

    # Model parameters
    parser.add_argument("--vocab_size", type = int, default = 10000)
    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--num_layers", type = int, default = 4)
    parser.add_argument("--d_model", type = int, default = 512)
    parser.add_argument("--num_heads", type = int, default = 16)
    parser.add_argument("--d_ff", type = int, default=1344, help="Defaults to 4*d_model if None")
    parser.add_argument("--rope_theta", type = float, default = 10000.0)
    parser.add_argument("--epsilon", type = float, default = 1e-8)

    # paramerters for generate_tokens()
    parser.add_argument("--max_new_tokens", type = int, default = 250)
    parser.add_argument("--temperature", type= float, default = 0.8)
    parser.add_argument("--top_p", type= float, default = 0.95)

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])


    args = parser.parse_args()


    # device checks
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")

    device = args.device


    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=[args.special_tokens])

    transformer_model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.rope_theta, args.epsilon, device = device, dtype = torch.float32)
    transformer_model.to(device)

    # dont really need to store iteration but this is what it returns
    _ = load_checkpoint(args.checkpoint_path, transformer_model, load_optimizer=False, optimizer=None)


    if args.user_text is None:
        print("Enter prompt, then Ctrl-D (Unix) / Ctrl-Z+Enter (Windows):")
        try:
            args.prompt = sys.stdin.read()
        except KeyboardInterrupt:
            args.user_text = ""


    prompt_ids = tokenizer.encode(args.user_text)

    # keep within context window (conservative: ensure room for at least 1 new token)
    if len(prompt_ids) >= args.context_length:
        prompt_ids = prompt_ids[-(args.context_length - 1):]

    prompt_ids_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)       # needs to be in 1-d longtensor for generate_tkns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # EOS id (if present)
    try:
        eos_id = tokenizer.bytes_to_id[args.special_tokens.encode("utf-8")]
    except KeyError:
        eos_id = -1  # won't trigger early stop

    with torch.no_grad():
        generated_ids = generate_tkns(transformer_model, prompt_ids_tensor, eos_id, args.max_new_tokens, args.temperature , args.top_p , device)
        
    generated_text = tokenizer.decode(generated_ids)

    print("\n=== Generation ===")
    print(generated_text)




if __name__ == "__main__":
    main()