from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.bases import AuxiliaryTask, SelfSupervisedModel
from models.supervised.classifier import Classifier

from models.tasks.patch_location import PatchLocationTask
from models.tasks.patch_shuffling import PatchShufflingTask
from models.tasks.reconstruction import VAEReconstructionTask
from models.tasks.rotation import RotationTask

from models.unsupervised.generative_model import GenerativeModel
from models.unsupervised.vae import VAE
from options import BaseOptions


@dataclass
class Options(BaseOptions):
    """ Set of options for the VAE MNIST Example. """
    hidden_size: int = 100  # dimensions of the hidden state (encoder output).
    learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

    # Prevent gradients of the classifier from backpropagating into the encoder.
    detach_classifier: bool = True
    
    # Settings for the reconstruction auxiliary task.
    reconstruction: VAEReconstructionTask.Options = mutable_field(VAEReconstructionTask.Options, coefficient=0.01)


class SelfSupervisedClassifier(nn.Module, Classifier, SelfSupervisedModel, GenerativeModel):
    def __init__(self, options: Options, num_classes: int = 10):
        super().__init__()
        Classifier.__init__(self, num_classes=num_classes)
        
        self.options: Options = options
        self.num_classes = num_classes
        self.hidden_size = options.hidden_size
        
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, self.hidden_size),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.classification_loss = nn.CrossEntropyLoss()

        self.tasks: List[AuxiliaryTask] = nn.ModuleList()  # type: ignore
        

        self.code_size = options.reconstruction.code_size
        self.reconstruction_task = VAEReconstructionTask(
            encoder=self.encoder,
            classifier=self.classifier,
            options=self.options.reconstruction,
            hidden_size=self.hidden_size,
        )
        self.tasks.append(self.reconstruction_task)

        self.optimizer =  optim.Adam(self.parameters(), lr=1e-3)
        self.device = self.options.device

    def get_loss(self, x: Tensor, y: Tensor=None) -> Tensor:
        loss = torch.zeros(1)
        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        if y is not None:
            supervised_loss = self.classification_loss(y_pred, y)
            loss += supervised_loss
        
        for task in self.tasks:
            task_loss = task.get_loss(x, h_x=h_x, y_pred=y_pred, y=y)
            loss += task_loss

        return loss

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        return self.reconstruction_task.get_loss(x=x, h_x=h_x)

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return Classifier.get_loss(self, x, y)

    def encode(self, x: Tensor):
        x = self.preprocess_inputs(x)
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        return x.view([x.shape[0], -1])

    def logits(self, h_x: Tensor) -> Tensor:
        if self.options.detach_classifier:
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
