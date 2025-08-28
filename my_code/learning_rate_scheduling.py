import math

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    """
    Cosine learning-rate schedule with linear warmup, per the handout.

    Args:
        t         : current iteration 
        alpha_max : peak learning rate reached at the end of warmup (and start of cosine)
        alpha_min : final learning rate after cosine annealing (held constant afterward)
        Tw        : number of warmup iterations 
        Tc        : last iteration index that is inside cosine-anneal phase
                    (inclusive). Must satisfy Tc >= Tw.

    Returns learning rate for iteration t
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    if Tc < Tw:
        raise ValueError("Tc must be >= Tw")

    # Handout: if t < Tw, alpha_t = (t / Tw) * alpha_max
    # Edge case: Tw == 0 -> skip warmup entirely.
    if Tw > 0 and t < Tw:
        return (t / Tw) * alpha_max

    # Cosine annealing: Tw <= t <= Tc 
    # alpha_t = alpha_min + 0.5*(1 + cos((t-Tw)/(Tc-Tw) * pi)) * (alpha_max - alpha_min)
    if t <= Tc:
        # If Tc == Tw, there is no interval so return a_max
        if Tc == Tw:
            return alpha_max



        val = (t - Tw) / (Tc - Tw)             
        return alpha_min + 0.5 * (1.0 + math.cos(math.pi * val)) * (alpha_max - alpha_min)

    #Post annealing: t > Tc -> return alpha_min 
    return alpha_min