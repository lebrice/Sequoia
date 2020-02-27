from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Generic, TypeVar

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from dataclasses import dataclass

from tasks import AuxiliaryTask
from models.bases import Model, BaseHParams
from config import Config


class SemiSupervisedModel(Model):
    def __init__(self, hparams: BaseHParams, config: Config):
        super().__init__(hparams, config)

    @abstractmethod
    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return SupervisedModel.get_loss(self, x, y)  # type: ignore
    
    @abstractmethod
    def unsupervised_loss(self, x: Tensor) -> Tensor:
        return UnsupervisedModel.get_loss(self, x)  # type: ignore

    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor=None):
        loss = self.unsupervised_loss(x)
        if y is not None:
            loss += self.supervised_loss(x, y)
        return loss


class SelfSupervisedModel(SemiSupervisedModel):
    def __init__(self, hparams: BaseHParams, config: Config):
        super().__init__(hparams, config)
        self.tasks: List[AuxiliaryTask] = nn.ModuleList()  # type: ignore
    
    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor=None):
        loss = self.unsupervised_loss(x)

        if y is not None:
            loss += self.supervised_loss(x, y)

        for task in self.tasks:
            loss += task.get_loss(x, y=y)

        return loss
