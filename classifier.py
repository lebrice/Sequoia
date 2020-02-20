from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.functional as F
from torch import Tensor, nn



from dataclasses import dataclass


class ImageNetHParams:
    pass

@orion.bootstrap_prior("imagenet-classification-256x256")
@orion.save_prior_as("simple-classifier-256x256")
@dataclass
class MyCustomHyperParameters(ImageNetHParams):
    batch_size: int = 32

    learning_rate: float = field(0.001, prior=Uniform(0.01, 1.0, trainable=True))

    num_layers: int = 10





class Classifier(nn.Module, ABC):
    def __init__(self):
        super().__init__(bob=123)
        
        self.feature_extractor: nn.Module = NotImplemented
        self.classifier: nn.Module = NotImplemented

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, logits=False) -> Tensor:  # type: ignore
        x = self.preprocess_inputs(x)
        h_x = self.extract_features(x)
        logits = self.logits(h_x)
        return h_x, logits
        if logits:
            return logits
        return self.log_probabilities(logits)


    @abstractmethod
    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        """ Extracts the features from input example `x`. """
        return self.feature_extractor(x)
    
    def logits(self, h_x: Tensor) -> Tensor:
        """ Returns the (raw) scores for each class given features `h_x`. """
        return self.classifier(h_x)
 
    def log_probabilities(self, logits: Tensor) -> Tensor:
        """ Returns the log-probabilities for each class given raw logits. """
        return self.log_softmax(logits)

    def probabilities(self, logits: Tensor) -> Tensor:
        """ Returns the probabilities for each class given input raw logits. """
        return self.log_probabilities(logits).exp()