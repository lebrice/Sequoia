import copy
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, List, NamedTuple, Optional, Tuple, Type,
                    TypeVar, Union, Callable)

import torch
from simple_parsing import MutableField as mutable_field
from simple_parsing import choice, field, list_field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from .CNN13 import CNN13
from torchvision.utils import save_image

from common.layers import ConvBlock, Flatten
from common.losses import LossInfo
from common.metrics import accuracy, get_metrics
from common.task import Task
from config import Config
from models.output_head import OutputHead
from tasks import AuxiliaryTask, AuxiliaryTaskOptions, Tasks
from utils.json_utils import JsonSerializable
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

        # Hyperparameters of the "output head" module.
        output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

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
            "cnn13": CNN13,
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
                 hparams: HParams,
                 config: Config):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Feature extractor
        self.encoder = encoder
        # Classifier output layer
        self.hparams: Classifier.HParams = hparams
        self.config = config
        self.logger = self.config.get_logger(__file__)

        self.hidden_size = hparams.hidden_size  
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = self.config.device

        # Classes of the current "task".
        # By default, contains all the classes in `range(0, self.num_classes)`.
        # When using a multihead approach (e.g. EWC), set `current_task` to the
        # classes found within the current task when training or evaluating.
        # NOTE: Order matters: task (0, 1) is not the same as (1, 0) (for now)
        # TODO: Replace the multiple classifier heads with something like CN-DPM so we can actually do task-free CL.
        self._default_task = Task(classes=list(range(self.num_classes)))
        self._current_task = self._default_task
        
        # Dictionary that maps from task classes to output head to be used.
        # By default, contains a single output head that serves all classes.
        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()  # type: ignore 
         # Classifier for the default task.

        self.output_heads[self._current_task.dumps()] = OutputHead(
            input_size=self.hidden_size,
            output_size=self.num_classes,
            hparams=self.hparams.output_head,
        )

        self.logger.info(f"output heads: {self.output_heads}")

        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.classifier # TODO: Also update this class attribute when switching tasks. 
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.hparams.aux_tasks.create_tasks(
            input_shape=input_shape,
            hidden_size=self.hparams.hidden_size
        )

        
        if self.config.debug and self.config.verbose:
            self.logger.debug(self)
            self.logger.debug("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                self.logger.debug(f"{task.name}: {task.coefficient}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  

    def supervised_loss(self, x: Tensor, y: Tensor, h_x: Tensor=None, y_pred: Tensor=None, loss_f: Callable[[Callable, Tensor],Tensor] = None) -> LossInfo:
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        y = y.view(-1)
        if loss_f is None:
            loss = self.classification_loss(y_pred, y)
            metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        else:
            loss = loss_f(self.classification_loss,y_pred)
        metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        loss_info = LossInfo(
            name=Tasks.SUPERVISED,
            total_loss=loss,
            tensors=(dict(x=x, h_x=h_x, y_pred=y_pred, y=y)),
        )
        loss_info.metrics[Tasks.SUPERVISED] = metrics
        return loss_info

    def get_loss(self, x: Tensor, y: Tensor=None, name: str="") -> LossInfo:
        if y is not None and y.shape[0] != x.shape[0]:
            raise RuntimeError("Whole batch can either be fully labeled or "
                               "fully unlabeled, but not a mix of both (for now)")
        total_loss = LossInfo(name)
        x, y = self.preprocess_inputs(x, y)
        h_x = self.encode(x)
        y_pred = self.logits(h_x)
        
        total_loss.total_loss = torch.zeros(1, device=self.device)
        total_loss.tensors["x"] = x.detach()
        total_loss.tensors["h_x"] = h_x.detach()
        total_loss.tensors["y_pred"] = y_pred.detach()

        # TODO: [improvement] Support a mix of labeled / unlabeled data.
        if y is not None:
            supervised_loss = self.supervised_loss(x=x, y=y, h_x=h_x, y_pred=y_pred)
            total_loss += supervised_loss

        for task_name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task_loss = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                total_loss += aux_task_loss
        
        if self.config.debug and self.config.verbose:
            for name, loss in total_loss.losses.items():
                self.logger.debug(name, loss.total_loss, loss.metrics)
        return total_loss

    def encode(self, x: Tensor):
        x, _ = self.preprocess_inputs(x, None)
        return self.encoder(x)

    def preprocess_inputs(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Optional[Tensor]]:
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
        # Process 'x'

        if x.shape[1:] != self.input_shape:
            x = fix_channels(x)
        
        if y is not None:
            # y_unique are the (sorted) unique values found within the batch.
            # idx[i] holds the index of the value at y[i] in y_unique, s.t. for
            # all i in range(0, len(y)) --> y[i] == y_unique[idx[i]]
            y_unique, idx = y.unique(sorted=True, return_inverse=True)
            # TODO: Could maybe decide which output head to use depending on the labels
            # (perhaps like the "labels trick" from https://arxiv.org/abs/1803.10123)
            if not (set(y_unique.tolist()) <= set(self.current_task.classes)):
                raise RuntimeError(
                    f"There are labels in the batch that aren't part of the "
                    f"current task! \n(Current task: {self.current_task}, "
                    f"batch labels: {y_unique})"
                )

            # NOTE: No need to do this when in the default task (all classes).
            if self.current_task != self._default_task:
                # if we are in the default task, no need to do this.
                # Re-label the given batch so the losses/metrics work correctly.
                new_y = torch.empty_like(y)
                for i, label in enumerate(self.current_task.classes):
                    new_y[y == label] = i
                y = new_y            
        return x, y

    def on_task_switch(self, task: Task, **kwargs) -> None:
        """Indicates to the model that it is working on a new task.

        Args:
            task_classes (Tuple[int, ...]): Tuple of integers, indicates the classes that are currently trained on.
        """
        self.current_task = task
        # also inform the auxiliary tasks that the task switched.
        for name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task.on_task_switch(task, **kwargs)

    def get_output_head(self, task: Task):
        return self.output_heads[task.dumps()]

    @property
    def classifier(self) -> nn.Module:
        return self.get_output_head(self.current_task)

    @property
    def current_task(self) -> Task:
        return self._current_task

    @current_task.setter
    def current_task(self, task: Task):
        assert isinstance(task, Task), f"Please set the current_task by passing a `Task` object."
        self._current_task = task
        
        task_str = task.dumps()
        # If there isn't an output head for this task
        if task_str not in self.output_heads:
            self.logger.debug(f"Creating a new output head for task {task}.")
            new_output_head = OutputHead(
                input_size=self.hidden_size,
                output_size=len(task.classes),
                hparams=self.hparams.output_head,
            ).to(self.device)
            self.output_heads[task_str] = new_output_head
            self.optimizer.add_param_group({"params": new_output_head.parameters()})

        # Update the classifier used by auxiliary tasks:
        AuxiliaryTask.classifier = self.classifier

    def logits(self, h_x: Tensor) -> Tensor:
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)
    
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool=False) -> Tuple[List[str], List[str]]:
        starting_task = self.current_task
        # Set the task ID attribute to create all the needed output heads. 
        for key in state_dict:
            if key.startswith("output_heads"):
                task_json_str = key.split(".")[1]
                task = Task.loads(task_json_str)
                # Set the task ID attribute to create all the needed output heads.
                self.current_task = task
                
        # Reset the task_id to the starting value.
        self.current_task = starting_task
        missing, unexpected = super().load_state_dict(state_dict, strict)
        # TODO: Make sure the mean-encoder and mean-output-head modules are loaded property when using Mixup.
        return missing, unexpected
    
    def optimizer_step(self, global_step: int, **kwargs) -> None:
        """Updates the model by calling `self.optimizer.step()`.
        Additionally, also informs the auxiliary tasks that the model got
        updated.
        """
        self.optimizer.step()
        for name, task in self.tasks.items():
            if task.enabled:
                task.on_model_changed(global_step=global_step, **kwargs)