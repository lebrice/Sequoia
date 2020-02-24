from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Classifier(ABC):
    def __init__(self, num_classes: int):
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
    