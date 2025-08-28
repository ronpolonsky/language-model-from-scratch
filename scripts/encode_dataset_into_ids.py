#!/usr/bin/env python3
import argparse
import os
import sys
import time
from array import array
from pathlib import Path
import numpy as np

from cs336_basics.tokenizer import Tokenizer  # <- your Tokenizer above

SPECIAL = "<|endoftext|>"

FILES = [
    ("data/TinyStoriesV2-GPT4-train.txt", "tinystories_train"),
    ("data/TinyStoriesV2-GPT4-valid.txt", "tinystories_valid"),
    # Add more if present:
    ("data/tinystories.txt",               "tinystories_all"),
    ("data/owt_train.txt",                 "owt_train"),
    ("data/owt_valid.txt",                 "owt_valid"),
]

def human(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def encode_one(in_path: str, out_dir: str, tok: Tokenizer, overwrite: bool=False) -> Path:
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = next(s for p, s in FILES if p == str(in_path))
    out_bin = out_dir / f"{stem}_ids_uint16.bin"
    done    = out_dir / f"{stem}_ids_uint16.bin.done"

    if out_bin.exists() and done.exists() and not overwrite:
        n = out_bin.stat().st_size // np.dtype(np.uint16).itemsize
        print(f"[skip] {in_path.name} -> {out_bin.name} (len≈{n:,})")
        return out_bin

    tmp = out_bin.with_suffix(out_bin.suffix + ".part")
    if tmp.exists():
        tmp.unlink(missing_ok=True)

    total_bytes = in_path.stat().st_size
    print(f"[encode] {in_path.name}  ({total_bytes/1e6:.1f} MB) -> {out_bin.name}")

    REPORT_MB = 8
    FLUSH_EVERY = 1_000_000

    t0 = time.time()
    bytes_read = 0
    toks_written = 0
    last_bucket = -1
    buf = array("H")

    try:
        with in_path.open("r", encoding="utf-8", errors="ignore") as fin, tmp.open("wb") as fout:
            for line in fin:
                bytes_read += len(line.encode("utf-8", "ignore"))
                ids = tok.encode(line)

                # uint16 bounds
                for tid in ids:
                    if tid >= 65536:
                        raise ValueError(f"token id {tid} doesn't fit uint16; increase dtype if needed.")

                buf.extend(ids)
                toks_written += len(ids)

                if len(buf) >= FLUSH_EVERY:
                    buf.tofile(fout)
                    buf = array("H")

                bucket = bytes_read // (REPORT_MB * 1_000_000)
                if bucket > last_bucket:
                    last_bucket = bucket
                    elapsed = time.time() - t0
                    mb = bytes_read / 1e6
                    mbps = mb / max(elapsed, 1e-6)
                    pct = 100.0 * bytes_read / max(total_bytes, 1)
                    eta = (total_bytes/1e6 - mb) / max(mbps, 1e-6)
                    print(f"[prog] {pct:5.1f}%  {mb:9.1f}/{total_bytes/1e6:.1f} MB  "
                          f"tokens={toks_written:,}  {mbps:5.1f} MB/s  ETA={human(eta)}")

            if buf:
                buf.tofile(fout)

        os.replace(tmp, out_bin)  # atomic finalize
        done.touch()
        n = out_bin.stat().st_size // np.dtype(np.uint16).itemsize
        print(f"[done]  {in_path.name} -> {out_bin.name}  tokens={toks_written:,}  secs={time.time()-t0:.1f}  len≈{n:,}")
        return out_bin

    except Exception:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        raise

def main():
    ap = argparse.ArgumentParser(description="Encode TinyStories/OWT .txt into uint16 token ids (streaming).")
    ap.add_argument("--vocab",  required=True, help="Path to vocab.tsv")
    ap.add_argument("--merges", required=True, help="Path to merges.txt")
    ap.add_argument("--outdir", default="data/encoded", help="Output directory")
    ap.add_argument("--only", choices=["train","valid"], help="Encode just one (TinyStories) file")
    ap.add_argument("--overwrite", action="store_true", help="Force re-encode")
    args = ap.parse_args()

    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[SPECIAL])

    to_do = []
    if args.only == "train":
        to_do = [("data/TinyStoriesV2-GPT4-train.txt", "tinystories_train")]
    elif args.only == "valid":
        to_do = [("data/TinyStoriesV2-GPT4-valid.txt", "tinystories_valid")]
    else:
        to_do = [(p, s) for p, s in FILES if Path(p).exists()]

    if not to_do:
        print("No known input files found. Expected one of:\n  " + "\n  ".join(p for p, _ in FILES))
        sys.exit(1)

    for p, _ in to_do:
        encode_one(p, args.outdir, tok, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
