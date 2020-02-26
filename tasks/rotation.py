from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bases import AuxiliaryTask


def rotate(x: Tensor, angle: float) -> Tensor:
    raise NotImplementedError("TODO")


class RotationTask(AuxiliaryTask):
    def get_loss(self,
                 x: Tensor,
                 h_x: Tensor,
                 y_pred: Tensor,
                 y: Tensor=None) -> Tensor:
        return torch.zeros(1)
        raise NotImplementedError("TODO")