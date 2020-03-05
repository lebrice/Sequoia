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
from common.losses import LossInfo
from common.metrics import Metrics, accuracy
from common.layers import ConvBlock, Flatten


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
        hidden_size: int = 32

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
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = self.config.device

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def get_loss(self, x: Tensor, y: Tensor, h_x: Tensor=None, y_pred: Tensor=None) -> LossInfo:
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        loss = self.classification_loss(y_pred, y)
        return LossInfo(
            total_loss=loss,
            tensors=(dict(x=x, h_x=h_x, y_pred=y_pred, y=y) if self.config.debug else {}),
            metrics=Metrics(x=x, h_x=h_x, y_pred=y_pred, y=y),
        )

    def encode(self, x: Tensor):
        x = self.preprocess_inputs(x)
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        """Preprocess the input tensor x before it is passed to the encoder.
        
        By default this does nothing. When subclassing the Classifier or 
        switching datasets, you might want to change this behaviour.

        Parameters
        ----------
        - x : Tensor
        
            a batch of inputs.
        
        Returns
        -------
        Tensor
            The preprocessed inputs.
        """
        return x

    def logits(self, h_x: Tensor) -> Tensor:
        return self.classifier(h_x)


class MnistClassifier(Classifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        self.hidden_size = hparams.hidden_size
        encoder = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, padding=1),
            ConvBlock(16, 32, kernel_size=3, padding=1),
            ConvBlock(32, self.hidden_size, kernel_size=3, padding=1),
            ConvBlock(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
        )
        classifier = nn.Sequential(
            Flatten(),
            nn.Linear(self.hidden_size, 10),
        )
        super().__init__(
            input_shape=(1,28,28),
            num_classes=10,
            encoder=encoder,
            classifier=classifier,
            hparams=hparams,
            config=config,
        )
    
    def preprocess_inputs(self, x):
        return super().preprocess_inputs(x)
