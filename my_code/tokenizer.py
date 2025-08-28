import regex as re
from pathlib import Path
from collections.abc import Iterable, Iterator


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # GPT-2-style
TOKEN_RE = re.compile(PATTERN)  # FIX: precompile for speed
BYTE = tuple(bytes([i]) for i in range(256))

class Tokenizer():

    def __init__(self, vocab, merges, special_tokens=None):

        self.merges = list(merges)
        self.pair_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.special_strs = set()
        self.SPECIALS_RE  = None

        # encoding maps final byte-chunks to ids; this must be O(1).
        self.id_to_bytes = dict(vocab)
        self.bytes_to_id = {b: i for i, b in self.id_to_bytes.items()}    # basically invert keys and vals of the dict


        if special_tokens is not None:
            specials_sorted = sorted(special_tokens, key=len, reverse=True)


            for s in specials_sorted:

                b = s.encode("utf-8")
                if b not in self.bytes_to_id:
                    new_id = (max(self.id_to_bytes) + 1) if self.id_to_bytes else 0

                    # need to add to vocab and the inverse convertor (need the utf8 for this)
                    self.id_to_bytes[new_id] = b 
                    self.bytes_to_id[b] = new_id

                self.special_strs.add(s)
            
            pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
            self.SPECIALS_RE = re.compile(pattern)




    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        """Load a Tokenizer from disk 
            Remember that in our other method "save_outputs we saved those files in this format:
            123\t746865 which we want to capture as tid_str="123", hex_bytes="746865
        """
        vpath = Path(vocab_filepath)
        mpath = Path(merges_filepath)

        # vocab.tsv: "<id>\t<bytes-hex>"
        vocab: dict[int, bytes] = {}
        with vpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tid_str, hex_bytes = line.split("\t")
                vocab[int(tid_str)] = bytes.fromhex(hex_bytes)

        # merges.txt: "<a-hex> <b-hex>" per line, in creation order
        merges: list[tuple[bytes, bytes]] = []
        with mpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a_hex, b_hex = line.split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))

        return cls(vocab, merges, special_tokens=special_tokens)




    # helper for encode()
    def _merge_bytes(self, seq: list[bytes]) -> list[bytes]:
        n = len(seq)
        if n < 2:
            return seq
        while True:
            best_rank = None
            best_pos = -1
            for i in range(n - 1):
                r = self.pair_rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
            if best_rank is None:
                break
            seq[best_pos:best_pos + 2] = [seq[best_pos] + seq[best_pos + 1]]
            n -= 1
        return seq


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """

        out_ids = []
        if self.SPECIALS_RE is not None:
            parts = self.SPECIALS_RE.split(text) 

        else:
            parts = [text]


        for part in parts:
            # for example iterate on parts of [ "",     "<|endoftext|>",       "world!!! kaka"  ]

            # skip if special or empty
            if not part:
                continue
            
            if part in self.special_strs:
                # emit the special token as a single id (atomic)
                out_ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue


            # We assume that a part in parts can be very long so use finditer()
            # For example, finditer() gives us this for the third part of the second split chunk.      
            # ["world",    "!!!",    " ",    "kaka"] 
            for m in re.finditer(PATTERN, part):

                # Here m would be a single "pre-token elment" (often single word) ---> "world", then "!!!", "kaka", .....
                # Note that per assignment handout, we DON'T merge across different m's (pre-tokens)!
                pretoken_str = m.group()

                # get it in sequence of bytes for the single element m
                pretoken_utf8 = pretoken_str.encode("utf-8")



                seq = [BYTE[b] for b in pretoken_utf8]
                seq = self._merge_bytes(seq)             # FAST greedy merge
                for tok_bytes in seq:
                    out_ids.append(self.bytes_to_id[tok_bytes])



        return out_ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Iterable is anything we can loop over.
        Lazily encode an iterable of text chunks (e.g., a file handle that yields lines).
        Yields token IDs in order, without loading the whole file into memory.
        """
        for chunk in iterable:
            if not chunk:
                continue
            # stream: process this chunk only; yield IDs as we go
            for tid in self.encode(chunk):
                yield tid



    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """


        # normalize to a Python list of ints
        if hasattr(ids, "tolist"):  # torch.Tensor / np.ndarray
            ids = ids.tolist()

        # Join all token bytes first 
        buf = b"".join(self.id_to_bytes[int(tid)] for tid in ids)  

        # UTF-8 is a self-describing, variable-length encoding, 
        # so a decoder does not need token boundaries. It just walks the byte stream 
        # left toright and, from the first byte of each character, knows how many bytes to consume
        out =  buf.decode("utf-8", errors="replace")
        return out