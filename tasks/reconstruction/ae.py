from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common.layers import DeConvBlock, Flatten, Reshape
from common.losses import LossInfo
from datasets.cifar import Cifar10, Cifar100
from datasets.dataset import DatasetConfig
from datasets.mnist import Mnist
from tasks.auxiliary_task import AuxiliaryTask

from .decoders import get_decoder

class AEReconstructionTask(AuxiliaryTask):
    """ Task that adds the AE loss (reconstruction loss). 
    
    Uses the feature extractor (`encoder`) of the parent model as the encoder of
    an AE. Contains trainable `decoder` module, which is
    used to get the AE loss to train the feature extractor with.
    """

    def __init__(self,
                 coefficient: float=None,
                 name: str="ae",
                 decoder: nn.Module=None,
                 options: "AEReconstructionTask.Options"=None):
        super().__init__(coefficient=coefficient, name=name, options=options)
        self.loss = nn.MSELoss(reduction="sum")
        self.decoder: nn.Module = get_decoder(
            input_size=AuxiliaryTask.input_shape,
            code_size=AuxiliaryTask.hidden_size,
        )

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> LossInfo:
        z = h_x.view([h_x.shape[0], -1])
        x_hat = self.decoder(z)

        recon_loss = self.reconstruction_loss(x_hat, x)

        loss_info = LossInfo(self.name, total_loss=recon_loss)
        return loss_info

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        z = h_x.view([h_x.shape[0], -1])
        x_hat = self.decoder(z)
        return x_hat

    def reconstruct(self, x: Tensor) -> Tensor:
        h_x = self.encode(x)
        x_hat = self.forward(h_x)
        return x_hat.view(x.shape)
    
    def reconstruction_loss(self, recon_x: Tensor, x: Tensor) -> Tensor:
        return self.loss(recon_x, x.view(recon_x.shape))
