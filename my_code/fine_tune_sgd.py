# fine_tune_sgd.py  (or keep your filename, just be consistent)
import torch, math
from typing import Optional, Callable

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, defaults={"lr": lr})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

def run_trial(lr, iters=10, seed=0):
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for _ in range(iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(float(loss))
        loss.backward()
        opt.step()
    return losses

if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        losses = run_trial(lr)
        print(f"lr={lr:.0e} -> {losses}")
