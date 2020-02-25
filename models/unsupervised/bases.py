from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Generic, TypeVar

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from dataclasses import dataclass

from models.bases import Model, BaseHParams


class UnsupervisedModel(Model, ABC):
    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        pass


class GenerativeModel(UnsupervisedModel, ABC):
    def __init__(self):
        super().__init__()

    def generate(self, z: Tensor) -> Tensor:
        pass

class AutoEncoder(UnsupervisedModel, ABC):
    @abstractmethod
    def encode(self, x: Tensor):
        pass
    
    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass

