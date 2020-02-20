
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.functional as F
from torch import Tensor, nn



class VAE(ABC, nn.Module):
    """ Example of a VAE:
    
    Taken from https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @abstractmethod
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
