import regex as re
import os
from collections import Counter, defaultdict
from my_code.pretokenization_example import find_chunk_boundaries


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""     # GPT-2 style pattern from the spec.


BYTE = tuple(bytes([i]) for i in range(256))  # reused 1-byte objects for efficiency in the helper below


#Efficiently converting "world" -> (w, o, r, l, d) but in its bytes utf-8 encoding.
def split_pretoken_into_bytes(b: bytes) -> tuple[bytes, ...]:
    """b must be UTF-8 bytes for a single pre-token."""
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError("Function expects UTF-8 bytes")
    return tuple(BYTE[x] for x in b)


    """
    Do ONE merge (A,B)->AB using two passes:       PERHAPS BE MORE EFFICIENT WITH ONE PASS ?
      1) decrement all old pairs in affected sequences; build new sequences
      2) increment all new pairs; re-index memberships
    
    """

def apply_merge_step(
    best_pair: tuple[bytes, bytes],
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
    ):

    tkn1, tkn2 = best_pair

    affected_sequences = list(pair_to_seqs.pop(best_pair, ()))
    
    if not affected_sequences:
        return

    # store all modified new_seqs
    modified = []

    for seq in affected_sequences:
        seq_len = len(seq)
        curr_count = pretoken_counts[seq]

        # Decrement all pairs and then rebuild them with the new seq with the merged token

        # NOTE that I am using two passes and can be more efficient with only one pass?
        for i in range(seq_len - 1):
            curr_pair = (seq[i] , seq[i + 1])
            pair_counts[curr_pair] -= curr_count

            if pair_counts[curr_pair] < 0:
                raise AssertionError(f"pair_counts went negative for {curr_pair}")

            if pair_counts[curr_pair] == 0:
                del pair_counts[curr_pair]
            
            if curr_pair in pair_to_seqs:
                    sset = pair_to_seqs[curr_pair]
                    if seq in sset:
                        sset.remove(seq)
                        if not sset:
                            del pair_to_seqs[curr_pair]


        # Build new sequence with merged token

        new_seq = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == tkn1 and seq[i+1] == tkn2:
                new_seq.append(tkn1 + tkn2)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1

        new_seq = tuple(new_seq)
        modified.append( (new_seq, seq, curr_count) )

        
        # Now modified stores all new sequences that have been modified
        # so ebuild new sequences and new entires with new merged token and everyting that touches it
    for new_seq, old_seq, count in modified:

        pretoken_counts.pop(old_seq, None)   # remove old
        pretoken_counts[new_seq] += count    # update new

        for j in range(len(new_seq) - 1):
            pair = (new_seq[j], new_seq[j + 1])

            pair_counts[pair] += count

            # associate seq with pair
            if pair not in pair_to_seqs:
                pair_to_seqs[pair] = set()
            pair_to_seqs[pair].add(new_seq)
            

#NOTE: CAN MAINTAIN A MIN HEAP FOR BETTER RUNTIME! (BUT WHAT ABOUT SPACE COMPLEXITY?)
# Returns the best pair. Chooses the lexicographically greater pair when counts tie.
def choose_best_pair(pair_counts):
    if not pair_counts:
        return None

    best_pair = None
    best_count = -1  # any real count will be >= 0

    for pair, cnt in pair_counts.items():
        if cnt > best_count:
            best_count = cnt
            best_pair = pair
        elif cnt == best_count:
            # tie: prefer lexicographically greater pair
            if best_pair is None or pair > best_pair:
                best_pair = pair

    return best_pair





def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    init_size = 256 + len(special_tokens)  #initial size of vocab
    next_index = init_size
    vocab = {}
    merges: list[tuple[bytes, bytes]] = []

    # "smart dicts"
    pretoken_counts = Counter()
    pair_counts = Counter()
    pair_to_seqs = defaultdict(set)

    escaped_special_tokens = map(re.escape, special_tokens)  #if special tokens can overlap, need to add sorting and store from longest to shortest before split
    splitting_pattern  = "|".join(escaped_special_tokens)


    #init  0....to...255 bytes
    for i in range(256):
        vocab[i] = bytes([i])   # for example bytes([ ... ,.... ,...]) creates a sequence of 3 bytes 

    #init special tokens
    for indx, tok in enumerate(special_tokens, start = 256):
        vocab[indx] = tok.encode("utf-8")  # --> already wrapped with bytes(...)


    # EXAMPLE DATA HERE IS:
    #                           "hello<|endoftext|>world!!! kaka<|endoftext|>bye"


    # Get chunks instead of our large corpus
    # In our example chunks are:

    # Chunk1: "hello"         Chunk2: "<|endoftext|>world!!! kaka"             Chunk3: "<|endoftext|>bye"
    with open(input_path, "rb") as f:
        num_processes = 4
        if special_tokens:
            split_tok = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_tok)
        
        else:
            # treat whole file as one chunk
            boundaries = [0, os.fstat(f.fileno()).st_size]

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)

            curr_chunk = f.read(end - start).decode("utf-8", errors="ignore")  

            parts = re.split(splitting_pattern, curr_chunk)

            # Eventually we get these parts:

            # Chunk1:  ["hello"]        Chunk2: ["", "<|endoftext|>", "world!!! kaka"]            Chunk3: ["", "<|endoftext|>", "bye"]

            # we never merge parts of one part with the other WITHIN EACH CHUNK
            # (There is no way for bytes from the special to be merged to bytes from "world!!! kaka")



            for part in parts:
                # for example iterate on parts of [  "",       "<|endoftext|>",      "world!!! kaka"   ]


                # skip if special or empty
                if not part or part in special_tokens:
                    continue
                
                # We assume that a part in parts can be very long so use finditer()
                # For example, finditer() gives us this for the third part of the second split chunk.      
                # ["world", "!!!", " ", "kaka"] 
                for m in re.finditer(PATTERN, part):

                    # Here m would be a single "pre-token elment" (often single word) ---> "world", then "!!!", "kaka", .....
                    # Note that per assignment handout, we DON'T merge across different m's (pre-tokens)!
                    pretoken_str = m.group()

                    # get it in sequence of bytes for the single element m
                    pretoken_utf8 = pretoken_str.encode("utf-8")

                    # Put this pre-token (m) into the dictionary (the key is the pre-token but split into its bytes)
                    # dict[ (w, o, r, l, d)] = count

                    pretoken_bytes = split_pretoken_into_bytes(pretoken_utf8)
                    
                    pretoken_counts[pretoken_bytes] += 1


    
    # Now we can initialize our counter of the tuple of consecutive bytes within each pre-token
    for sequence_of_bytes, seq_count in pretoken_counts.items():
        for i in range(len(sequence_of_bytes) - 1):
            # sequences of length < 2 naturally contribute no pairs.

            pair = (sequence_of_bytes[i] , sequence_of_bytes[i + 1])
            pair_counts[pair] += seq_count
            pair_to_seqs[pair].add(sequence_of_bytes)

        
    # Now we have:
    # pretoken_counts = {  (w, o, r, l, d) : 3,    (!, !, ! ): 1,    etc..... }
    # pair_counts =    {   (w, o): 1,     (o, r) : 1,              etc...   }
    # pair_to_seqs =    {  sequences that currently contain each one of the pairs.  }
    # vocab = { ....... }




    # assume I can pop the best
    while next_index < vocab_size and pair_counts:

        best_pair = choose_best_pair(pair_counts)

        # check we actually got a legit pair (even though I delete keys with count 0)
        if best_pair is None or pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        apply_merge_step(best_pair, pretoken_counts, pair_counts, pair_to_seqs)

        # add to vocab
        vocab[next_index] = best_pair[0] + best_pair[1]
        next_index += 1

    return vocab, merges
