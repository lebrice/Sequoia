from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .task import Task
from models.unsupervised.autoencoder import AutoEncoder


class ReconstructionTask(Task):
    def __init__(self, model: AutoEncoder):
        self.model = model

    def get_loss(self, x: Tensor, h_x: Tensor) -> Tensor:
        x_hat = self.model.decode(h_x)
        return F.binary_cross_entropy(x, x_hat, reduction="sum")

        

    