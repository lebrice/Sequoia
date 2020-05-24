import copy
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type,
                    TypeVar, Union)

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import choice, field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image

from common.layers import ConvBlock, Flatten
from common.losses import LossInfo
from common.metrics import accuracy, get_metrics
from config import Config
from tasks import AuxiliaryTask, AuxiliaryTaskOptions, Tasks
from utils.utils import fix_channels

logger = logging.getLogger(__file__)

class Classifier(nn.Module):
    @dataclass
    class HParams:
        """ Set of hyperparameters for the classifier.

        We use [simple_parsing](www.github.com/lebrice/simpleparsing) to
        generate command-line arguments for each attribute of this class.
        """
        batch_size: int = 128   # Input batch size for training.
        epochs: int = 10        # Number of epochs to train.
        learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

        # Dimensions of the hidden state (feature extractor/encoder output).
        hidden_size: int = 100

        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False

        # Use an encoder architecture from the torchvision.models package.
        encoder_model: Optional[str] = choice({
            "vgg16": models.vgg16,  # This is the only one tested so far.
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "alexnet": models.alexnet,
            # "squeezenet": models.squeezenet1_0,  # Not supported yet (weird output shape)
            "densenet": models.densenet161,
            # "inception": models.inception_v3,  # Not supported yet (creating model takes forever?)
            # "googlenet": models.googlenet,  # Not supported yet (creating model takes forever?)
            # "shufflenet": models.shufflenet_v2_x1_0,
            # "mobilenet": models.mobilenet_v2,
            # "resnext50_32x4d": models.resnext50_32x4d,
            # "wide_resnet50_2": models.wide_resnet50_2,
            # "mnasnet": models.mnasnet1_0,
        }, default=None)
        # Use the pretrained weights of the ImageNet model from torchvision.
        pretrained_model: bool = False
        # Freeze the weights of the pretrained encoder (except the last layer,
        # which projects from their hidden size to ours).
        freeze_pretrained_model: bool = False


        aux_tasks: AuxiliaryTaskOptions = field(default_factory=AuxiliaryTaskOptions)

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 encoder: nn.Module,
                 classifier: nn.Module,
                #  auxiliary_task_options: AuxiliaryTaskOptions,
                 hparams: HParams,
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Feature extractor
        self.encoder = encoder
        # Classifier output layer
        self._classifier = classifier
        self.hparams: Classifier.HParams = hparams
        self.config = config

        self.hidden_size = hparams.hidden_size  
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = self.config.device

        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self._classifier
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.hparams.aux_tasks.create_tasks(  # type: ignore
            input_shape=input_shape,
            hidden_size=self.hparams.hidden_size
        )

        # Current task label. (Optional, as we shouldn't rely on this.)
        # TODO: Replace the classifier model with something like CN-DPM or CURL,
        # so we can actually do task-free CL.
        self._current_task_id: Optional[str] = None
        # Dictionary of classifiers to use if we are provided the task-label.
        self.task_classifiers: Dict[str, nn.Module] = nn.ModuleDict()  #type: ignore  

        if self.config.debug and self.config.verbose:
            logger.debug(self)
            logger.debug("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                logger.debug(f"{task.name}: {task.coefficient}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  

    def supervised_loss(self, x: Tensor, y: Tensor, h_x: Tensor=None, y_pred: Tensor=None) -> LossInfo:
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        y = y.view(-1)
        loss = self.classification_loss(y_pred, y)
        metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        loss_info = LossInfo(
            name=Tasks.SUPERVISED,
            total_loss=loss,
            tensors=(dict(x=x, h_x=h_x, y_pred=y_pred, y=y)),
        )
        loss_info.metrics[Tasks.SUPERVISED] = metrics
        return loss_info

    def get_loss(self, x: Tensor, y: Tensor=None) -> LossInfo:
        total_loss = LossInfo("Train" if self.training else "Test")
        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        total_loss.total_loss = torch.zeros(1, device=self.device)
        total_loss.tensors["x"] = x.detach()
        total_loss.tensors["h_x"] = h_x.detach()
        total_loss.tensors["y_pred"] = y_pred.detach()

        if y is not None:
            supervised_loss = self.supervised_loss(x=x[:len(y)], y=y, h_x=h_x[:len(y)], y_pred=y_pred[:len(y)])
            total_loss += supervised_loss

        for task_name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task_loss = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                total_loss += aux_task_loss
        
        if self.config.debug and self.config.verbose:
            for name, loss in total_loss.losses.items():
                logger.debug(name, loss.total_loss, loss.metrics)
        return total_loss


    def encode(self, x: Tensor):
        x = self.preprocess_inputs(x)
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor) -> Tensor:
        """Preprocess the input tensor x before it is passed to the encoder.
        
        By default this does nothing. When subclassing the Classifier or 
        switching datasets, you might want to change this behaviour.

        Parameters
        ----------
        - x : Tensor
        
            a batch of inputs.
        
        Returns
        -------
        Tensor
            The preprocessed inputs.
        """
        return fix_channels(x)

    def on_task_switch(self, task_id: Optional[Union[int, str]]) -> None:
        if isinstance(task_id, int):
            task_id = str(task_id)
        self.current_task_id = task_id
        # also inform the auxiliary tasks that the task switched.
        for name, task in self.tasks.items():
            task.on_task_switch(task_id=task_id)

    @property
    def classifier(self) -> nn.Module:
        if self.current_task_id is None:
            return self._classifier
        else:
            return self.task_classifiers[self.current_task_id]

    @property
    def current_task_id(self) -> Optional[str]:
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: Optional[Union[int, str]]):
        value_str: Optional[str] = str(value) if isinstance(value, int) else value
        self._current_task_id = value_str
        # If there isn't a classifier for this task
        if value_str and value_str not in self.task_classifiers.keys():
            if self.config.debug:
                logger.info(f"Creating a new classifier for taskid {value}.")
            # Create one starting from the "global" classifier.
            classifier = copy.deepcopy(self._classifier)
            self.task_classifiers[value_str] = classifier
            self.optimizer.add_param_group({"params": classifier.parameters()})

    def logits(self, h_x: Tensor) -> Tensor:
        if self.hparams.detach_classifier:
            h_x = h_x.detach()

        # Use the "general" classifier by default.
        classifier = self.classifier
        # If a task-id is given, use the task-specific classifier.
        if self.current_task_id is not None:
            classifier = self.task_classifiers[self.current_task_id]
        return classifier(h_x)
    
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool=True) -> Tuple[List[str], List[str]]:
        starting_task_id = self.current_task_id
        # Set the task ID attribute to create all the needed output heads. 
        for key in state_dict:
            if key.startswith("task_classifiers"):
                task_id = key.split(".")[1]
                self.on_task_switch(task_id)
        # Reset the task_id to the starting value.
        self.on_task_switch(starting_task_id)
        return super().load_state_dict(state_dict, strict)
    
    def optimizer_step(self, global_step: int) -> None:
        """Updates the model by calling `self.optimizer.step()`.
        Additionally, also informs the auxiliary tasks that the model got
        updated.
        """
        self.optimizer.step()
        for name, task in self.tasks.items():
            task.on_model_changed(global_step=global_step) 
