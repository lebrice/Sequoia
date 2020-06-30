from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common.layers import DeConvBlock, Flatten, Reshape
from common.losses import LossInfo
from tasks.auxiliary_task import AuxiliaryTask

from .decoder_for_dataset import get_decoder_class_for_dataset

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
        
        # TODO: This is weird, should be able to use the dataset here, no?
        decoder_class = get_decoder_class_for_dataset(AuxiliaryTask.input_shape)
        self.decoder: nn.Module = None

    def enable(self):
        self.decoder: nn.Module = self.decoder_class(
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
