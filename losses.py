from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.functional as F
from torch import Tensor, nn

class SemiSupervisedLoss(ABC):
    @abstractmethod
    def loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor) -> Tensor:
        pass

class MixupLoss(SemiSupervisedLoss):
    def __init__(self, parent: nn.Module):
        self.parent = parent
    
    def loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor) -> Tensor:
        
