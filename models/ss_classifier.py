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
from models.classifier import Classifier
from tasks import AuxiliaryTask, VAEReconstructionTask

class SelfSupervisedClassifier(Classifier):
    @dataclass
    class HParams(Classifier.HParams):
        """ Set of Options / Command-line Parameters for the MNIST Example.
        
        We use [simple_parsing](www.github.com/lebrice/simpleparsing) to generate
        all the command-line arguments for this class.
        """
        # Dimensions of the hidden state (feature extractor/encoder output).
        hidden_size: int = 100

        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False

        reconstruction = VAEReconstructionTask.Options(coefficient=1e-3)

    def __init__(self, tasks: List[AuxiliaryTask], hparams: HParams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
        self.reconstruction_task = VAEReconstructionTask(options=self.hparams.reconstruction)

        # Share the relevant parameters with all the auxiliary tasks.
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs

    def get_loss(self, x: Tensor, y: Tensor=None) -> LossInfo:
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

    def unsupervised_loss(self, x: Tensor, h_x: Tensor=None, y_pred: Tensor=None) -> LossInfo:
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
