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
from tasks import AuxiliaryTask, AuxiliaryTaskOptions


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

        # Dimensions of the hidden state (feature extractor/encoder output).
        hidden_size: int = 100

        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False

        aux_tasks: AuxiliaryTaskOptions = field(default_factory=AuxiliaryTaskOptions)

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 encoder: nn.Module,
                 classifier: nn.Module,
                #  auxiliary_task_options: AuxiliaryTaskOptions,
                 hparams: HParams,
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Feature extractor
        self.encoder = encoder
        # Classifier output layer
        self.classifier = classifier
        self.hparams: Classifier.HParams = hparams
        self.config = config

        self.hidden_size = hparams.hidden_size  
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = self.config.device

        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting a class attribute.
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        
        self.tasks: Dict[str, AuxiliaryTask] = self.hparams.aux_tasks.create_tasks(  # type: ignore
            input_shape=input_shape,
            hidden_size=self.hparams.hidden_size
        )

        if self.config.debug:
            print(self)
            print("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                print(f"{task.name}: {task.coefficient}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)        

    def supervised_loss(self, x: Tensor, y: Tensor, h_x: Tensor=None, y_pred: Tensor=None) -> LossInfo:
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        y = y.view(-1)
        loss = self.classification_loss(y_pred, y)
        return LossInfo(
            total_loss=loss,
            tensors=(dict(x=x, h_x=h_x, y_pred=y_pred, y=y)),
            metrics=Metrics(x=x, h_x=h_x, y_pred=y_pred, y=y),
        )

    def get_loss(self, x: Tensor, y: Tensor=None) -> LossInfo:
        loss_info = LossInfo()

        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        loss_info.total_loss = torch.zeros(1, device=self.device)
        loss_info.tensors["h_x"] = h_x
        loss_info.tensors["y_pred"] = y_pred

        if y is not None:
            supervised_loss = self.supervised_loss(x=x, y=y, h_x=h_x, y_pred=y_pred)
            loss_info.losses["supervised"] = supervised_loss.total_loss
            loss_info += supervised_loss

        for task_name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task_loss = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                if self.config.verbose:
                    print(f"{aux_task.name}:\t {aux_task_loss.total_loss.item()}")
                loss_info += aux_task_loss
        
        return loss_info

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
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)
