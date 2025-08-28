import math
from typing import Optional, Callable
import torch


class AdamWOptimizer(torch.optim.Optimizer):
    r"""
    Hand-rolled AdamW (Loshchilov & Hutter, 2019), matching the handout.

    Per-parameter state:
      - exp_avg:   first moment (m),    same shape as parameter
      - exp_avg_sq: second moment (v),  same shape as parameter
      - step:      integer iteration counter (t), starts at 0; we use (t+1) as the 1-based step

    Update (for each parameter theta with grad g):
        m     = b1 m + (1 - b1) g
        v     = b2 v + (1 - b2) g * g
        t     = t + 1
        a_t   = a * sqrt(1 - b2^t) / (1 - b1^t)               # bias-corrected step size
        theta =   theta - a_t * m / (sqrt(v) + aps)                   # Adam step (no decay mixed into g)
        theta =   theta - a * lmbda * theta                                 # decoupled weight decay
    """

    def __init__(self, params, lr, betas, eps, weight_decay):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")



        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)




    def set_lr(self, lr: float):
        """
            When need to update lr mid model training (due to cosine annealing for example)
        """
        for g in self.param_groups:
            g["lr"] = lr


    @torch.no_grad()
    def step(self,  closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:

            # acees defaults that are copied to every group's dictionary when the group is created.
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # get the current parameters estimates
            for param in group["params"]:
                if param.grad is None:
                    continue


                gradient = param.grad

                # each nn.Parameter in my model (model.parameters) gets its own Optimizer state entry.
                # HOWEVER, this self.state[param] is inhereted from torch....Optimizer as just an empty dict 
                # --> So I need to poppulate it with variables that I care about adn will use.
                # So here,  get the variables of the optimizer's state for this underlyig nn.Parameter param:
                param_state  = self.state[param]

                if len(param_state) == 0:
                    # first time we 'stepped' into param and so need to store those variables we will use and care about.
                    param_state["t"] = 0
                    param_state["m"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    param_state["v"] = torch.zeros_like(param, memory_format=torch.preserve_format)


                # access its value
                param_state["t"] += 1           # starts from 1 from the handout
                t = param_state["t"]


                m = param_state["m"]
                v = param_state["v"]

                
                # update the state of the parameter (Note that this is updated in place so there is also no need to manualy update param_state["m"] for example)

                m.mul_(beta1).add_(gradient, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)


                step_size = lr * math.sqrt(1.0 - (beta2 ** t)) / (1.0 - (beta1 ** t) )

                # # Adam step: thata = thata - a_t * m / (sqrt(v) + eps)
                param.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                
                # apply weight decay
                param.add_(param, alpha=-lr * weight_decay)

                # update state after iteration (already updated t before that)

        
                
        return loss  