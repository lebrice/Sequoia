"""
Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
"""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import grad

from .bases import AuxiliaryTask


class IrmTask(AuxiliaryTask):
    """IRM implementation taken from https://github.com/facebookresearch/InvariantRiskMinimization/blob/6aad47e689913b9bdad05880833530a5edac389e/code/colored_mnist/main.py#L107
        
    TODO: Understand what's happening here.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_nll = nn.CrossEntropyLoss()

    def penalty(self, logits: Tensor, y: Tensor) -> Tensor:
        scale = torch.ones([1], requires_grad=True).to(logits.device)
        loss = self.mean_nll(logits * scale, y)
        grads = grad(loss, [scale], create_graph=True)[0]
        return (grads**2).sum()

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
        if y is None:
            return torch.zeros(1)
        y = y.to(y_pred.device)
        return self.penalty(y_pred, y)
    

def irm_demo():
    """ Demo code from the IRM paper. """
    def compute_penalty(losses, dummy_w ):
        g1 = grad(losses[0::2].mean(), dummy_w, create_graph = True)[0]
        g2 = grad(losses[1::2].mean(), dummy_w, create_graph = True)[0]
        return (g1 * g2).sum()

    def example_1(n =10000 , d=2, env =1):
        x1 = torch.randn(n, d) * env
        y = x1 + torch.randn(n, d) * env
        x2 = y + torch.randn(n, d)
        return torch.cat((x1, x2), 1), y.sum(1, keepdim = True)

    phi = torch.ones(4, 1, requires_grad=True)
    dummy_w = torch.ones(1, requires_grad=True)
    opt = torch.optim.SGD([phi], lr=1e-3)
    mse = torch.nn.MSELoss(reduction="none")

    environments = [example_1(env=0.1) , example_1(env=1.0)]

    for iteration in range (50000):
        error = torch.zeros(1)
        penalty = torch.zeros(1)
        for x_e, y_e in environments:
            p = torch.randperm(len(x_e))
            error_e = mse(x_e[p] @ phi * dummy_w , y_e [p])
            penalty += compute_penalty(error_e, dummy_w)
            error += error_e.mean()
        
        opt.zero_grad()
        (1e-5 * error + penalty).backward()
        opt.step()
        if iteration % 1000 == 0:
            print(phi.numpy())



if __name__ == "__main__":
    irm_demo()