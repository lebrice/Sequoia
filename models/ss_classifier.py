from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Type, TypeVar

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from common.losses import LossInfo
from config import Config
from models.classifier import Classifier
from tasks import (AuxiliaryTask, IrmTask, JigsawPuzzleTask, ManifoldMixupTask,
                   MixupTask, PatchLocationTask, RotationTask, TaskType,
                   VAEReconstructionTask, AdjustBrightnessTask)
from common.layers import Flatten, ConvBlock

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

        reconstruction: VAEReconstructionTask.Options = VAEReconstructionTask.Options(coefficient=1e-3)
        mixup:          MixupTask.Options             = MixupTask.Options(coefficient=1e-3)
        manifold_mixup: ManifoldMixupTask.Options     = ManifoldMixupTask.Options(coefficient=1e-3)
        rotation:       RotationTask.Options          = RotationTask.Options(coefficient=1e-3)
        jigsaw:         JigsawPuzzleTask.Options      = JigsawPuzzleTask.Options(coefficient=0)
        irm:            IrmTask.Options               = IrmTask.Options(coefficient=1e-3)
        adjust_brightness: AdjustBrightnessTask.Options = AdjustBrightnessTask.Options(coefficient=1e-3)

        def get_tasks(self) -> List[AuxiliaryTask]:
            tasks: List[AuxiliaryTask] = []
            tasks.append(VAEReconstructionTask(options=self.reconstruction))
            tasks.append(MixupTask(options=self.mixup))
            tasks.append(ManifoldMixupTask(options=self.manifold_mixup))
            tasks.append(RotationTask(options=self.rotation))
            tasks.append(JigsawPuzzleTask(options=self.jigsaw))
            tasks.append(IrmTask(options=self.irm))
            tasks.append(AdjustBrightnessTask(options=self.adjust_brightness))
            return tasks

    def __init__(self, hparams: HParams, *args, **kwargs):
        super().__init__(*args, hparams=hparams, **kwargs)
        # Share the relevant parameters with all the auxiliary tasks.
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        # TODO: Share the hidden size dimensions of this model with the Auxiliary tasks so they know how big the h_x is going to actually be.
        AuxiliaryTask.hidden_size = self.hparams.hidden_size
        aux_tasks = self.hparams.get_tasks()
        self.tasks: List[AuxiliaryTask] = nn.ModuleList(aux_tasks)  # type: ignore
        if self.config.verbose:
            print(self)
            print("Auxiliary tasks:")
            for task in self.tasks:
                print(f"{task.name} - enabled: {task.enabled}, coefficient: {task.coefficient}")


    def get_loss(self, x: Tensor, y: Tensor=None) -> LossInfo:
        loss_info = LossInfo()

        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        loss_info.tensors["h_x"] = h_x
        loss_info.tensors["y_pred"] = y_pred

        if y is not None:
            supervised_loss = super().get_loss(x, y, h_x=h_x, y_pred=y_pred)
            loss_info.losses["supervised"] = supervised_loss.total_loss
            loss_info += supervised_loss

        for aux_task in self.tasks:
            if aux_task.enabled:
                aux_task_loss = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                if self.config.verbose:
                    print(f"{aux_task.name}:\t {aux_task_loss.total_loss.item()}")
                loss_info += aux_task_loss
        
        return loss_info

    def logits(self, h_x: Tensor) -> Tensor:
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)


from models.classifier import MnistClassifier as BaseMnistClassifier


class MnistClassifier(BaseMnistClassifier, SelfSupervisedClassifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        super().__init__(hparams=hparams, config=config)