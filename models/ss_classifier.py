from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Type, TypeVar, Optional

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
from datasets.dataset import DatasetConfig

@dataclass
class AuxiliaryTaskList(nn.ModuleList, List[AuxiliaryTask]):  #type: ignore
    """ TODO: maybe use this in order to have the same hyperparameter class as `classifier`. """
    reconstruction: VAEReconstructionTask.Options = VAEReconstructionTask.Options(coefficient=0.)
    mixup:          MixupTask.Options             = MixupTask.Options(coefficient=0.)
    manifold_mixup: ManifoldMixupTask.Options     = ManifoldMixupTask.Options(coefficient=0.)
    rotation:       RotationTask.Options          = RotationTask.Options(coefficient=0.)
    jigsaw:         JigsawPuzzleTask.Options      = JigsawPuzzleTask.Options(coefficient=0.)
    irm:            IrmTask.Options               = IrmTask.Options(coefficient=0.)
    adjust_brightness: AdjustBrightnessTask.Options = AdjustBrightnessTask.Options(coefficient=0.)

    def __post_init__(self):
        # Call the __init__ of the ModuleList base class.
        super().__init__()

    def create_tasks(self, input_shape: Tuple[int, ...], hidden_size: int) -> None:
        # Set the class attributes so that the Tasks can be created.
        AuxiliaryTask.hidden_size = hidden_size
        AuxiliaryTask.input_shape = input_shape

        self.append(VAEReconstructionTask(options=self.reconstruction))
        self.append(MixupTask(options=self.mixup))
        self.append(ManifoldMixupTask(options=self.manifold_mixup))
        self.append(RotationTask(options=self.rotation))
        self.append(JigsawPuzzleTask(options=self.jigsaw))
        self.append(IrmTask(options=self.irm))
        self.append(AdjustBrightnessTask(options=self.adjust_brightness))


class SelfSupervisedClassifier(Classifier):

    @dataclass
    class HParams(Classifier.HParams):
        aux_tasks: AuxiliaryTaskList = field(default_factory=AuxiliaryTaskList)
        
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 hparams: HParams,
                 *args, **kwargs):
        super().__init__(*args, input_shape=input_shape, hparams=hparams, **kwargs)

        # Share the relevant parameters with all the auxiliary tasks.
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs

        self.tasks: AuxiliaryTaskList = self.hparams.aux_tasks
        self.tasks.create_tasks(input_shape=input_shape, hidden_size=self.hparams.hidden_size)
        self.tasks = nn.ModuleList(self.tasks)
        if self.config.debug:
            print(self)
            print("Auxiliary tasks:")
            print(self.tasks)
            for task in self.tasks:
                print(f"{task.name}: {task.coefficient}")


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
