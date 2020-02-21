from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Classifier(nn.Module, ABC):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        self.feature_extractor: nn.Module = NotImplemented
        self.classifier: nn.Module = NotImplemented       

    def get_loss(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.extract_features(x)
        logits = self.logits(h_x)
        y_hat = self.log_probabilities(logits)
        return F.NLLLoss(y_hat, y)

    @abstractmethod
    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        """ Extracts the features from input example `x`. """
        return self.feature_extractor(x)
    
    @abstractmethod
    def logits(self, h_x: Tensor) -> Tensor:
        """ Returns the (raw) scores for each class given features `h_x`. """
        return self.classifier(h_x)
 
    def log_probabilities(self, logits: Tensor) -> Tensor:
        """ Returns the log-probabilities for each class given raw logits. """
        return F.log_softmax(logits)

    def probabilities(self, logits: Tensor) -> Tensor:
        """ Returns the probabilities for each class given input raw logits. """
        return self.log_probabilities(logits).exp()