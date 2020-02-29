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
from tasks import AuxiliaryTask
from models.common import LossInfo

@dataclass
class HParams:
    """ Set of Options / Command-line Parameters for the MNIST Example.
    
    We use [simple_parsing](www.github.com/lebrice/simpleparsing) to generate
    all the command-line arguments for this class.
    
    """
    batch_size: int = 128   # Input batch size for training.
    epochs: int = 10        # Number of epochs to train.
    learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

    # Dimensions of the hidden state (encoder output).
    hidden_size: int = 100

    # Prevent gradients of the classifier from backpropagating into the encoder.
    detach_classifier: bool = False




class SelfSupervisedClassifier(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 hparams: HParams,
                 tasks: List[AuxiliaryTask],
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.hparams = hparams
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
        self.config = config

        self.hidden_size = hparams.hidden_size
        self.num_classes = num_classes
        
        
        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, self.hidden_size),
            nn.Sigmoid(),
        )
        # Classifier output layer
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.classification_loss = nn.CrossEntropyLoss()

        
        # Share the relevant parameters with all the auxiliary tasks.
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs

        self.optimizer =  optim.Adam(self.parameters(), lr=1e-3)
        self.device = self.config.device


    def get_loss(self, x: Tensor, y: Tensor=None) -> LossInfo:
        # TODO: return logs
        loss_tuple = LossInfo()

        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        loss_tuple.tensors["h_x"] = h_x
        loss_tuple.tensors["y_pred"] = y_pred

        if y is not None:
            loss_tuple += self.supervised_loss(y_pred, y)

        for aux_task in self.tasks:
            if aux_task.enabled:
                loss_tuple += aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y) 

        return loss_tuple

    def unsupervised_loss(self, x: Tensor, h_x: Tensor=None, y_pred: Tensor=None) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        losses = LossInfo()
        raise NotImplementedError("TODO")

    def supervised_loss(self, y_pred: Tensor, y: Tensor) -> LossInfo:
        supervised_loss = self.classification_loss(y_pred, y)
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
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)

    def reconstruct(self, x: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        x_hat = self.reconstruction_task(h_x)
        return x_hat.view(x.shape)
    
    def generate(self, z: Tensor) -> Tensor:
        z = z.to(self.device)
        return self.reconstruction_task(z)
