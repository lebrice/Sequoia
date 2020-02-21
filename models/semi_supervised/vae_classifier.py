from abc import ABC, abstractmethod
from typing import Any, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import SelfSupervisedModel, AuxiliaryTask
from models.supervised.classifier import Classifier
from models.unsupervised.vae import VAE

from models.tasks.reconstruction import VAEReconstructionTask
        

class VaeClassifier(nn.Module, Classifier, SelfSupervisedModel):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_size: int = 20,
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

        tasks = tasks or []
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
        self.detach_representation_from_classifier: bool = True

        self.tasks.append(VAEReconstructionTask(code_size=hidden_size))


    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        loss = 0.
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        y_logits = self.logits(h_x)
        if y is not None:
            loss += Classifier.classification_loss(y_logits, y)
        
        # reconstruction loss:
        for task in self.tasks:
            task_loss = task.get_loss(x, h_x=h_x, y_pred=y_logits, y=y)
            print(f"Task {type(task)} has a loss of {task_loss}")
            loss += task_loss
        return loss

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        return self.vae.get_loss(x)

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
