"""
Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)

TODO: Refactor / Validate / test this Task in the new setup.
"""
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import grad

from .auxiliary_task import AuxiliaryTask
from sequoia.common.loss import Loss

class IrmTask(AuxiliaryTask):
    """IRM implementation taken from https://github.com/facebookresearch/InvariantRiskMinimization/blob/6aad47e689913b9bdad05880833530a5edac389e/code/colored_mnist/main.py#L107
        
    TODO: Understand what's happening here.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_nll = nn.CrossEntropyLoss()

    def penalty(self, logits: Tensor, y: Tensor) -> Tensor:
        y = y.to(logits.device)
        scale = torch.ones([1], requires_grad=True, device=logits.device)
        loss = self.mean_nll(logits * scale, y)
        grads = grad(loss, scale, create_graph=True)[0]
        return (grads**2).sum()

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Loss:
        if y is None:
            return Loss()
        if not y_pred.requires_grad:
            # Can't evaluate the IRM score when the y_pred doesn't require grad!
            with torch.enable_grad():
                y_pred = self.classifier(h_x)
                loss = self.penalty(y_pred, y)
                return Loss(loss)
        loss = self.penalty(y_pred, y)
        return Loss(loss)
    

def irm_demo():
    """ Demo code from the IRM paper. """
    def compute_penalty(losses: Tensor, dummy_w: Tensor) -> Tensor:
        loss_1 = losses[0::2].mean()
        loss_2 = losses[1::2].mean()
        g1 = grad(loss_1, dummy_w, create_graph = True)[0]
        g2 = grad(loss_2, dummy_w, create_graph = True)[0]
        return (g1 * g2).sum()

    def example_1(n =10000 , d=2, env =1):
        x1 = torch.randn(n, d) * env
        y = x1 + torch.randn(n, d) * env
        x2 = y + torch.randn(n, d)
        labels = y.sum(1, keepdim = True)
        return torch.cat((x1, x2), 1), labels

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
            
            x_e_shuffled = x_e[p]
            y_e_shuffled = y_e[p]

            error_e = mse(dummy_w * x_e_shuffled @ phi  , y_e_shuffled)
            
            penalty += compute_penalty(error_e, dummy_w)
            error += error_e.mean()
        
        opt.zero_grad()
        (1e-5 * error + penalty).backward()
        opt.step()
        if iteration % 1000 == 0:
            print(phi)



if __name__ == "__main__":
    irm_demo()