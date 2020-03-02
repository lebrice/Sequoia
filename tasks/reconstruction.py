from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bases import AuxiliaryTask
from common.losses import LossInfo

class VAEReconstructionTask(AuxiliaryTask):
    """ Task that adds the VAE loss (reconstruction + KL divergence). 
    
    Uses the feature extractor (`encoder`) of the parent model as the encoder of
    a VAE. Contains trainable `mu`, `logvar`, and `decoder` modules, which are
    used to get the VAE loss to train the feature extractor with.    
    """
        
    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Settings & Hyper-parameters related to the VAEReconstructionTask. """
        code_size: int = 50  # dimensions of the VAE code-space.

    def __init__(self, coefficient: float=None, options: "VAEReconstructionTask.Options"=None):
        super().__init__(coefficient=coefficient, options=options)
        self.code_size = self.options.code_size
        # add the rest of the VAE layers: (Mu, Sigma, and the decoder)
        self.mu     = nn.Linear(AuxiliaryTask.hidden_size, self.code_size)
        self.logvar = nn.Linear(AuxiliaryTask.hidden_size, self.code_size)
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

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> LossInfo:
        h_x = h_x.view([h_x.shape[0], -1])
        mu, logvar = self.mu(h_x), self.logvar(h_x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        
        recon_loss = self.reconstruction_loss(x_hat, x)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        loss = recon_loss + kl_loss
        loss_info = LossInfo()
        loss_info.total_loss = loss
        loss_info.losses["reconstruction"] = recon_loss
        loss_info.losses["kl"] = kl_loss
        return loss_info

    def reconstruct(self, x: Tensor) -> Tensor:  
        h_x = self.encode(x)
        x_hat = self.forward(h_x)
        return x_hat.view(x.shape)
    
    def generate(self, z: Tensor) -> Tensor:
        z = z.to(self.device)
        return self.forward(z)

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
        

    