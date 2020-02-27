from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Tuple, TypeVar

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from config import Config
from models.bases import BaseHParams, Model


class SupervisedModel(Model):
    def __init__(self, hparams: BaseHParams, config: Config):
        super().__init__(hparams, config)

    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor) -> Tensor:
        pass


class Classifier(SupervisedModel):
    def __init__(self, hparams: BaseHParams, config: Config, num_classes: int):
        super().__init__(hparams, config)
        self.num_classes = num_classes
        self.encoder: nn.Module = NotImplemented
        self.classifier: nn.Module = NotImplemented
        self.loss = nn.CrossEntropyLoss()  

    def get_loss(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        logits = self.logits(h_x)
        return self.loss(logits, y)

    @abstractmethod
    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """ Extracts the features from input example `x`. """
        return self.encoder(x)

    @abstractmethod
    def logits(self, h_x: Tensor) -> Tensor:
        """ Returns the (raw) scores for each class given features `h_x`. """
        return self.classifier(h_x)
 
    def probabilities(self, logits: Tensor) -> Tensor:
        """ Returns the probabilities for each class given input raw logits. """
        return F.softmax(logits)
