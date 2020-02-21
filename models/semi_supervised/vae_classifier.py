from abc import ABC, abstractmethod
from typing import Any, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import SelfSupervisedModel, AuxiliaryTask
from models.supervised.classifier import Classifier
from models.unsupervised.vae import VAE


class VaeClassifier(nn.Module, Classifier, SelfSupervisedModel):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_size: int = 20,
                 tasks: List[AuxiliaryTask]=None):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        super().__init__()
        Classifier.__init__(self, num_classes=num_classes)

        self.vae = VAE(code_size=self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )

        tasks = tasks or []
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(tasks)  # type: ignore
        self.detach_representation_from_classifier: bool = True

    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        loss = 0.
        x = self.preprocess_inputs(x)
        mu, logvar = self.vae.encode(x)
        h_x = self.vae.reparameterize(mu, logvar)

        # Reconstruction loss:
        x_hat = self.vae.decode(h_x)
        loss += self.vae.reconstruction_loss(x_hat, x)
        
        # VAE loss:
        loss += self.vae.kl_divergence_loss(mu, logvar)

        y_logits = self.logits(h_x)
        if y is not None:
            loss += Classifier.classification_loss(y_logits, y)
        
        # reconstruction loss:
        for task in self.tasks:
            loss += task.get_loss(x, x_hat=x_hat, y_pred=y_logits, y=y)

        return loss

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        return VAE.get_loss(self, x)

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return Classifier.get_loss(self, x, y)

    def extract_features(self, x: Tensor):
        mu, logvar = self.vae.encode(x)
        h_x = self.vae.reparameterize(mu, logvar)
        return h_x

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def logits(self, h_x: Tensor) -> Tensor:
        if self.detach_representation_from_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)
