from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import AuxiliaryTask
from models.unsupervised.autoencoder import AutoEncoder


class VAEReconstructionTask(nn.Module, AuxiliaryTask):
    def __init__(self, code_size: int = 100):
        self.code_size = code_size
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(self.code_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> Tensor:
        code_size = h_x.shape[-1]
        mu, logvar = h_x[..., :code_size//2], h_x[..., code_size//2:]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        loss = self.reconstruction_loss(recon_x, x)
        loss += self.kl_divergence_loss(mu, logvar)
        print("Auxiliary loss (VAE):", loss.item())
        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def reconstruction_loss(recon_x: Tensor, x: Tensor) -> Tensor:
        return F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    @staticmethod
    def kl_divergence_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        

    