import os
import argparse
import time
import platform
import resource
from pathlib import Path
from my_code.bpe_tokenizer import train_bpe  

SPECIAL = "<|endoftext|>"




def save_outputs(out_dir: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:

    """ 
    Save the output to a dir 
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "vocab.tsv").open("w", encoding="utf-8") as f:
        for tid in sorted(vocab):
            f.write(f"{tid}\t{vocab[tid].hex()}\n")
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")



def show_utf8(b: bytes) -> str:
    """ 
    For showing the longest token
    """
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        # fall back to latin-1 (reversible) so you can still eyeball it
        return repr(b.decode("latin-1"))


def peak_mem_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes; Linux returns KB
    return (ru.ru_maxrss / (1024**3)) if platform.system() == "Darwin" else ((ru.ru_maxrss * 1024) / (1024**3))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/tinystories.txt"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/tinystories_bpe"))
    ap.add_argument("--vocab_size", type=int, default=10_000)
    args = ap.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(str(args.corpus), args.vocab_size, special_tokens=[SPECIAL])
    wall_s = time.perf_counter() - start  # record how much time it took
    mem_gb = peak_mem_gb()                # record memory usage

    save_outputs(args.out_dir, vocab, merges)


    # longest token
    longest_id = max(vocab, key=lambda i: len(vocab[i]))
    longest = vocab[longest_id]
    L = len(longest)

    # 1-2 sentence deliverable
    print(
        f"Trained byte-level BPE (vocab=10,000 incl. {SPECIAL}) on TinyStories; "
        f"training took {wall_s/3600:.2f} hours and ~{mem_gb:.2f} GB peak RAM. "
        f"Longest token is {L} bytes: {show_utf8(longest)}.")




if __name__ == "__main__":
    main()