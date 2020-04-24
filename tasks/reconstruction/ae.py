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
from tasks.reconstruction.vae import MnistDecoder, CifarDecoder


class AEReconstructionTask(AuxiliaryTask):
    """ Task that adds the AE loss (reconstruction loss). 
    
    Uses the feature extractor (`encoder`) of the parent model as the encoder of
    an AE. Contains trainable `decoder` module, which is
    used to get the AE loss to train the feature extractor with.
    """
    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Settings & Hyper-parameters related to the VAEReconstructionTask. """
        code_size: int = 50  # dimensions of the AE code-space.

    def __init__(self,
                 coefficient: float=None,
                 name: str="ae",
                 options: "AEReconstructionTask.Options"=None):
        super().__init__(coefficient=coefficient, name=name, options=options)
        self.options: AEReconstructionTask.Options
        self.code_size = self.options.code_size  # type: ignore
        
        self.decoder: nn.Module
        if AuxiliaryTask.input_shape == Mnist.x_shape:
            # TODO: get the right decoder architecture for other datasets than MNIST.
            self.decoder = MnistDecoder(code_size=self.code_size)
        elif AuxiliaryTask.input_shape == Cifar10.x_shape:
            self.decoder = CifarDecoder(code_size=self.code_size)
        else:
            raise RuntimeError(f"Don't have an encoder for the given input shape: {AuxiliaryTask.input_shape}")

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        z = h_x.view([h_x.shape[0], -1])
        x_hat = self.decoder(z)
        return x_hat

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor=None, y: Tensor=None) -> LossInfo:
        z = h_x.view([h_x.shape[0], -1])
        x_hat = self.decoder(z)

        recon_loss = self.reconstruction_loss(x_hat, x)

        loss_info = LossInfo(self.name)
        loss_info += LossInfo("recon", total_loss=recon_loss)
        return loss_info

    def reconstruct(self, x: Tensor) -> Tensor:
        h_x = self.encode(x)
        x_hat = self.forward(h_x)
        return x_hat.view(x.shape)
    
    def generate(self, z: Tensor) -> Tensor:
        z = z.to(self.device)
        return self.forward(z)

    # Reconstruction loss summed over all elements and batch
    @staticmethod
    def reconstruction_loss(recon_x: Tensor, x: Tensor) -> Tensor:
        return F.mse_loss(recon_x, x.view(recon_x.shape), size_average=False)
