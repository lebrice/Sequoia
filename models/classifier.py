from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, TypeVar, NamedTuple

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import Config
from models.common import LossInfo




class Classifier(nn.Module):

    @dataclass
    class HParams:
        """ Set of hyperparameters for the classifier.

        We use [simple_parsing](www.github.com/lebrice/simpleparsing) to
        generate command-line arguments for each attribute of this class.
        """
        batch_size: int = 128   # Input batch size for training.
        epochs: int = 10        # Number of epochs to train.
        learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

        # Dimensions of the hidden state (encoder output).
        hidden_size: int = 100

        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False


    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 hparams: HParams,
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Feature extractor
        self.encoder = encoder
        # Classifier output layer
        self.classifier = classifier
        self.hparams = hparams
        self.config = config

        self.hidden_size = hparams.hidden_size  
        self.loss = nn.CrossEntropyLoss()
        self.device = self.config.device

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)



    def get_loss(self, x: Tensor, y: Tensor) -> LossInfo:
        # TODO: return logs
        loss_info = LossInfo()

        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        loss_info.tensors["h_x"] = h_x
        loss_info.tensors["y_pred"] = y_pred

        loss_info += self.supervised_loss(y_pred, y)

        return loss_info

    def supervised_loss(self, y_pred: Tensor, y: Tensor) -> LossInfo:
        supervised_loss = self.loss(y_pred, y)
        metrics = self.get_metrics(y_pred, y)
        loss_info = LossInfo(
            supervised_loss,
            losses={"supervised": supervised_loss},
            metrics=metrics
        )
        return loss_info


    def get_metrics(self, y_pred: Tensor, y: Tensor) -> Dict[str, Any]:
        #TODO: calculate accuracy or other metrics.
        batch_size = y_pred.shape[0]
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == y).sum() / batch_size
        return {
            "Accuracy": accuracy
        }

    def encode(self, x: Tensor):
        x = self.preprocess_inputs(x)
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def logits(self, h_x: Tensor) -> Tensor:
        return self.classifier(h_x)


class MnistClassifier(Classifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        self.hidden_size = hparams.hidden_size
        encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, self.hidden_size),
            nn.Sigmoid(),
        )
        classifier = nn.Linear(self.hidden_size, 10)
        super().__init__(
            input_shape=(1,28,28),
            num_classes=10,
            encoder=encoder,
            classifier=classifier,
            hparams=hparams,
            config=config,
        )