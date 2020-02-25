from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.bases import AuxiliaryTask, TaskOptions


class VAEReconstructionTask(AuxiliaryTask):
    """ Task that adds the VAE loss (reconstruction + KL divergence). """
        
    @dataclass
    class Options(TaskOptions):
        """ Settings & Hyper-parameters related to the VAEReconstructioTask. """
        code_size: int = 50  # dimensions of the VAE code-space.


    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 options: Options,
                 hidden_size: int):
        super().__init__(encoder=encoder,
                         classifier=classifier,
                         options=options)
        self.hidden_size = hidden_size
        self.code_size = options.code_size
        # We use the feature extractor as the encoder of a VAE.
        # add the rest of the VAE layers: (Mu, Sigma, and the decoder)
        self.mu =  nn.Linear(self.hidden_size, self.code_size)
        self.logvar = nn.Linear(self.hidden_size, self.code_size)
        self.decoder = nn.Sequential(
            nn.Linear(self.code_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        h_x = h_x.view([h_x.shape[0], -1])
        mu, logvar = self.mu(h_x), self.logvar(h_x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> Tensor:
        h_x = h_x.view([h_x.shape[0], -1])
        mu, logvar = self.mu(h_x), self.logvar(h_x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        loss = self.reconstruction_loss(x_hat, x)
        loss += self.kl_divergence_loss(mu, logvar)
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
        

    