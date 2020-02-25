from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Tuple, TypeVar

import torch
from simple_parsing import field
from simple_parsing.utils import JsonSerializable
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.tasks import AuxiliaryTask
from utils import cuda_available, gpus_available
from .config import Config

@dataclass
class BaseHParams(JsonSerializable):
    """ Set of hyperparameters for the VAE MNIST Example. """
    batch_size: int = 128   # Input batch size for training.
    epochs: int = 10        # Number of epochs to train.
    learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

class Model(nn.Module):
    def __init__(self, hparams: BaseHParams, config: Config):
        super().__init__()
        self.hparams = hparams
        self.config = config
    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor):
        pass
