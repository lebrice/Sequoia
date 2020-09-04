from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common.layers import DeConvBlock, Flatten, Reshape
from common.loss import Loss
from common.tasks.auxiliary_task import AuxiliaryTask

from .ae import AEReconstructionTask
from .decoder_for_dataset import get_decoder_class_for_dataset

class VAEReconstructionTask(AEReconstructionTask):
    """ Task that adds the VAE loss (reconstruction + KL divergence). 
    
    Uses the feature extractor (`encoder`) of the parent model as the encoder of
    a VAE. Contains trainable `mu`, `logvar`, and `decoder` modules, which are
    used to get the VAE loss to train the feature extractor with.    
    """
    name: ClassVar[str] = "vae"

    @dataclass
    class Options(AEReconstructionTask.Options):
        """ Settings & Hyper-parameters related to the VAEReconstructionTask. """
        code_size: int = 50  # dimensions of the VAE code-space.
        beta: float = 1.0  # Beta term, multiplies the KL divergence term.

    def __init__(self,
                 coefficient: float = None,
                 options: "VAEReconstructionTask.Options" = None):
        super().__init__(coefficient=coefficient, options=options)
        self.options: VAEReconstructionTask.Options
        self.code_size = self.options.code_size
        # add the rest of the VAE layers: (Mu, Sigma, and the decoder)
        self.mu     = nn.Linear(AuxiliaryTask.hidden_size, self.code_size)
        self.logvar = nn.Linear(AuxiliaryTask.hidden_size, self.code_size)
        decoder_class = get_decoder_class_for_dataset(AuxiliaryTask.input_shape)
        self.decoder: nn.Module = decoder_class(
            code_size=self.code_size,
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

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> Loss:
        h_x = h_x.view([h_x.shape[0], -1])
        mu, logvar = self.mu(h_x), self.logvar(h_x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        recon_loss = self.reconstruction_loss(x_hat, x)
        kl_loss = self.options.beta * self.kl_divergence_loss(mu, logvar)
        loss = Loss(self.name, tensors=dict(mu=mu, logvar=logvar, z=z, x_hat=x_hat))
        loss += Loss("recon", loss=recon_loss)
        loss += Loss("kl", loss=kl_loss)
        return loss

    def generate(self, z: Tensor) -> Tensor:
        z = z.to(self.device)
        return self.forward(z)

    @staticmethod
    def kl_divergence_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

