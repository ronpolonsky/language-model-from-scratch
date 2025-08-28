import torch


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save everything needed to resume training:
      - model weights (state_dict)
      - optimizer state (e.g., AdamW moments)
      - current iteration (for LR schedule / bookkeeping)
      - RNG states (optional but helpful for exact reproducibility)

    'out' can be a filesystem path or an already-opened binary file-like object.
    """

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(payload, out)




def load_checkpoint(src, model, load_optimizer: bool, optimizer= None):

    loaded_payload = torch.load(src, map_location="cpu")
    model.load_state_dict(loaded_payload["model_state"])
    
    if load_optimizer:
        if optimizer is None:
            raise ValueError("need to pass in optimizer to load its state")
        else:    
            optimizer.load_state_dict(loaded_payload["optimizer_state"])
        

    return int(loaded_payload.get("iteration",0))

