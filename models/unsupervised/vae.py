
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .autoencoder import AutoEncoder


class VAE(AutoEncoder):
    """ Example of a VAE for MNIST
    
    Adapted from https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    def __init__(self, code_size: int = 20):
        super().__init__()
        self.code_size: int = code_size
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.code_size)
        self.fc22 = nn.Linear(400, self.code_size)
        self.fc3 = nn.Linear(self.code_size, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def get_loss(self, x: Tensor) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        loss = self.reconstruction_loss(recon_x, x)
        loss += self.kl_divergence_loss(mu, logvar)
        return loss

    def decode(self, z: Tensor) -> Tensor:
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


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