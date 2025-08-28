import torch
from my_code.softmax_with_temperature import softmax_with_temp




@torch.no_grad()
def generate_tkns(model, prompt_ids, eos_token_id: int, max_new_tokens: int = 100, temperature: float = 1.0, top_p = None, device = None):
    """
    Autoregressive decoding for a sequence using temperature scaling and optional top-p sampling.

    Args:
        model: TransformerLM that maps (B, T) -> logits (B, T, V) in fp32.
        prompt_ids: 1D or 2D LongTensor of token ids. If 1D, treated as shape (T,).
        eos_token_id: int id of the end-of-sequence token; decoding stops when produced.
        max_new_tokens: maximum number of tokens to generate.
        temperature: tau for temperature scaling. tau=1 leaves logits unchanged; tau->0 becomes greedy.
        top_p: nucleus threshold in (0,1]. If None or >=1, no truncation .
        device: device to run on. If None, inferred from model parameters.

    Returns:
        LongTensor of shape (T_total,) if return_prompt else (gen_len,)
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure tensor, on device, and with a batch dim (B=1)
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

    if prompt_ids.ndim == 1:
        seq = prompt_ids.unsqueeze(0)  # (1, T)                   
    elif prompt_ids.ndim == 2 and prompt_ids.size(0) == 1:
        seq = prompt_ids
    else:
        raise ValueError("decode expects a single sequence: 1D or (1, T)")
    seq = seq.to(device=device, dtype=torch.long)                 


    model.eval()  # make sure we're in eval mode for generation
    
    generated = []

    for _ in range(max_new_tokens):
        logits = model(seq)                     # (B, T, V)
        last_logits = logits[:, -1, :]           # (1, V)


        if temperature is None or temperature <= 0:
            # just pick the one with the highest score
            next_id = torch.argmax(last_logits, dim=-1)  

        else:
            probs = softmax_with_temp(last_logits, temperature)
            
             # Nucleus (top-p) filtering
            if top_p is not None and 0.0 < top_p < 1.0:

                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  
                #take the top probs until exceed top_p
                cdf = torch.cumsum(sorted_probs, dim=-1) 

                keep = cdf <= top_p                                     # (B, V) boolean mask
                keep = torch.roll(keep, shifts=1, dims=-1)              # include the token that "crosses" top_p
                keep[..., 0] = True                                     # guarantees the top token is always kept so the set is never empty.
                

                # Zero out tail, then renormalize
                filtered_sorted = sorted_probs * keep

                denom = filtered_sorted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                filtered_sorted = filtered_sorted / denom

                # Scatter back to original index order
                filtered = torch.zeros_like(probs)
                probs = filtered.scatter(-1, sorted_idx, filtered_sorted)


            # Sample one token id from the categorical distribution
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)
        

        # next_id is a tensor of shape (1,) returned from multinomial, so need to convert to integer.
        token = int(next_id.item())
        generated.append(token)

        # append to the running sequence for the next step
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)   # FIX: keep shape (1, T+1)

         # Stop when EOS is generated
        if token == eos_token_id:
            break
        
    return torch.tensor(generated, dtype=torch.long, device = device)