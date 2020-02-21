from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GenerativeModel(ABC):
    def __init__(self):
        super().__init__()

    def generate(self, z: Tensor) -> Tensor:
        pass