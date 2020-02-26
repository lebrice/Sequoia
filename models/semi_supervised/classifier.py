from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, TypeVar

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.bases import BaseHParams
from models.config import Config
from models.supervised import Classifier
from models.unsupervised import GenerativeModel
from tasks import (TaskType, AuxiliaryTask, ManifoldMixupTask, MixupTask,
                   PatchLocationTask, JigsawPuzzleTask, RotationTask,
                   VAEReconstructionTask, IrmTask)

from .bases import SelfSupervisedModel


@dataclass
class HParams(BaseHParams):
    """ Set of Options / Command-line Parameters for the MNIST Example.
    
    We use [simple_parsing](www.github.com/lebrice/simpleparsing) to generate
    all the command-line arguments for this class.
    
    """
    # Dimensions of the hidden state (encoder output).
    hidden_size: int = 100

    # Prevent gradients of the classifier from backpropagating into the encoder.
    detach_classifier: bool = True

    # Settings for the reconstruction auxiliary task.
    reconstruction: VAEReconstructionTask.Options = VAEReconstructionTask.Options(coefficient=0.001)
    
    # Settings for the "vanilla" mixup auxiliary task.
    mixup:          MixupTask.Options = MixupTask.Options(coefficient=0.001)
    
    # Settings for the manifold mixup auxiliary task.
    manifold_mixup: ManifoldMixupTask.Options = ManifoldMixupTask.Options(coefficient=0.1)
    
    # Settings for the rotation auxiliary task.
    rotation:       RotationTask.Options = RotationTask.Options(coefficient=0)
    
    # Settings for the jigsaw puzzle auxiliary task.
    jigsaw:         JigsawPuzzleTask.Options = JigsawPuzzleTask.Options(coefficient=0)
    
    # Settings for the Invariant Risk Minimization auxiliary task.
    irm:            IrmTask.Options = IrmTask.Options(coefficient=1)


class SelfSupervisedClassifier(Classifier):
    def __init__(self, hparams: HParams, config: Config):
        super().__init__(hparams=hparams, config=config)

        self.hidden_size = hparams.hidden_size
        self.num_classes = config.num_classes
        
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

        self.tasks: List[AuxiliaryTask] = nn.ModuleList()  # type: ignore
        
        self.code_size = hparams.reconstruction.code_size
        
        # Reconstruction auxiliary task
        recon_task = self.add_task(VAEReconstructionTask,
            options=self.hparams.reconstruction,
            hidden_size=self.hidden_size,
        )
        self.reconstruction_task: VAEReconstructionTask = recon_task

        # Rotation detection task:
        self.add_task(RotationTask, options=self.hparams.rotation)
        
        # Jigsaw puzzle task:
        self.add_task(JigsawPuzzleTask, options=self.hparams.jigsaw)


        # Mixup and Manifold-Mixup Auxiliary Tasks:
        self.add_task(ManifoldMixupTask, options=self.hparams.manifold_mixup)
        self.add_task(MixupTask, options=self.hparams.mixup)
        
        self.add_task(IrmTask, options=self.hparams.irm)
                
        self.optimizer =  optim.Adam(self.parameters(), lr=1e-3)
        self.device = self.config.device


    def add_task(self,
                 task_type: Type[TaskType],
                 options: AuxiliaryTask.Options=None, **kwargs) -> TaskType:
        task = task_type(  # type: ignore
            encoder=self.encoder,
            classifier=self.classifier,
            preprocessing=self.preprocess_inputs,
            options=options, **kwargs
        )
        self.tasks.append(task)
        return task


    def get_loss(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Dict[str, Tensor]]:
        # TODO: return logs
        losses: Dict[str, Tensor] = OrderedDict()
        total_loss = torch.zeros(1)

        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        if y is not None:
            supervised_loss = self.classification_loss(y_pred, y)
            losses["supervised"] = supervised_loss
            total_loss += supervised_loss
        
        for aux_task in self.tasks:
            if aux_task.enabled:
                aux_loss = aux_task.get_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                aux_loss_scaled = aux_loss * aux_task.coefficient
                
                total_loss += aux_loss_scaled
                
                losses[f"{aux_task.name}"] = aux_loss
                losses[f"{aux_task.name}_scaled"] = aux_loss_scaled

        return total_loss, losses

    def unsupervised_loss(self, x: Tensor) -> Tensor:
        x = self.preprocess_inputs(x)
        h_x = self.encode(x)
        return self.reconstruction_task.get_loss(x=x, h_x=h_x)

    def supervised_loss(self, x: Tensor, y: Tensor) -> Tensor:
        total_loss, losses = self.get_loss(x, y)
        return losses["supervised"]
        return Classifier.get_loss(self, x, y)

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
