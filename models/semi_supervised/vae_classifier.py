from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from models.base import AuxiliaryTask, SelfSupervisedModel
from models.supervised.classifier import Classifier
from models.tasks.reconstruction import VAEReconstructionTask
from models.unsupervised.vae import VAE


@dataclass
class ModelHyperParameters:
    pass

class VaeClassifier(nn.Module, Classifier, SelfSupervisedModel):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_size: int = 100,
                 tasks: List[AuxiliaryTask]=None):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        super().__init__()
        Classifier.__init__(self, num_classes=num_classes)

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, self.hidden_size),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        tasks = tasks or []
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
        self.detach_representation_from_classifier: bool = True

        self.reconstruction_task = VAEReconstructionTask(code_size=hidden_size)
        self.tasks.append(self.reconstruction_task)


    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        loss = torch.zeros(1)
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        y_logits = self.logits(h_x)
        
        if y is not None:
            loss += self.classification_loss(y_logits, y)
        
        for task in self.tasks:
            task_loss = task.get_loss(x, h_x=h_x, y_pred=y_logits, y=y)
            loss += task_loss
        return loss

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        return self.reconstruction_task.get_loss(x=x, h_x=h_x)

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return Classifier.get_loss(self, x, y)

    def encode(self, x: Tensor):
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def logits(self, h_x: Tensor) -> Tensor:
        if self.detach_representation_from_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)
