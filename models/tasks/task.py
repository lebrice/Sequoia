from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, TypeVar, Generic

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from models.vae_classifier import Model

class Task(nn.Module, ABC):
    def __init__(self, model: Model):
        super().__init__()
        self.model: Model = model

    @abstractmethod
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor) -> Tensor:
        pass
