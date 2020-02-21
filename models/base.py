from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Generic, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Model(ABC):
    pass


class UnsupervisedModel(Model):
    @abstractmethod
    def get_loss(self, x: Tensor) -> Tensor:
        pass


class SupervisedModel(Model):
    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor) -> Tensor:
        pass


class SemiSupervisedModel(SupervisedModel, UnsupervisedModel, ABC):  # type: ignore
    @abstractmethod
    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return SupervisedModel.get_loss(self, x, y)
    
    @abstractmethod
    def unsupervised_loss(self, x: Tensor) -> Tensor:
        return UnsupervisedModel.get_loss(self, x)

    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor=None):
        loss = torch.zeros(1)
        loss += self.unsupervised_loss(x)
        if y is not None:
            loss += self.supervised_loss(x, y)
        return loss


class AuxiliaryTask(nn.Module, ABC):
    def __init__(self, model: "SelfSupervisedModel"):
        super().__init__()
        self.model: Model = model

    @abstractmethod
    def get_loss(self, x: Tensor, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None) -> Tensor:
        pass


class SelfSupervisedModel(SemiSupervisedModel):
    def __init__(self, tasks: List[AuxiliaryTask]=None):
        tasks = [] or tasks
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
    
    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor=None):
        loss = self.unsupervised_loss(x)

        if y is not None:
            loss += self.supervised_loss(x, y)

        for task in self.tasks:
            loss += task.get_loss(x, y=y)

        return loss
