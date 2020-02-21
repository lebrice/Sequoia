from abc import ABC, abstractmethod
from typing import Any, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import SemiSupervisedModel

from models.supervised.classifier import Classifier
from models.unsupervised.vae import VAE


class VaeClassifier(Classifier, VAE, SemiSupervisedModel):
    def __init__(self, num_classes: int = 10, hidden_size:int = 20):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        Classifier.__init__(self)
        VAE.__init__(self, code_size=hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )
        from tasks.reconstruction import ReconstructionTask
        from tasks.task import Task, UnsupervisedTask, SemiSupervisedTask
        self.tasks: List[Task] = nn.ModuleList()  # type: ignore
                
        self.detach_representation_from_classifier: bool = True

    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        loss = 0.
        
        x = self.preprocess_inputs(x)
        mu, logvar = self.encode(x)
        h_x = self.reparameterize(mu, logvar)

        x_hat = self.decode(h_x)
        
        # Reconstruction loss:
        # VAE loss:
        loss += VAE.reconstruction_loss(x_hat, x)
        loss += VAE.kl_divergence_loss(mu, logvar)

        y_logits = self.logits(h_x)
        if y is not None:
            loss += Classifier.classification_loss(y_logits, y)
        
        from tasks.task import UnsupervisedTask
        # reconstruction loss:
        for task in self.tasks:
            loss += task.get_loss()
        
        if y is not None:
            loss += self.supervised_loss()
        return loss

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        mu, logvar = self.encode(x)
        h_x = self.reparameterize(mu, logvar)


    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def logits(self, h_x: Tensor) -> Tensor:
        if self.detach_representation_from_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)
