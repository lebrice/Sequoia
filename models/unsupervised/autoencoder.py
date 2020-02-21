from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor, nn


class AutoEncoder(ABC):
    @abstractmethod
    def encode(self, x: Tensor):
        pass
    
    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass