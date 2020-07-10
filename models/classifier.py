import copy
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, NamedTuple, Optional,
                    Tuple, Type, TypeVar, Union)

import pytorch_lightning as pl
import torch
from pytorch_lightning import metrics
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from torchvision.utils import save_image

from common.layers import ConvBlock, Flatten
from common.losses import LossInfo
from common.metrics import Metrics, accuracy, get_metrics
from common.task import Task
from config import Config
from models.output_head import OutputHead
from pl_bolts.datamodules import LightningDataModule
from simple_parsing import MutableField as mutable_field
from simple_parsing import choice, field, list_field
from tasks import AuxiliaryTask, AuxiliaryTaskOptions, Tasks
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from utils.utils import add_prefix, fix_channels

from .pretrained_model import get_pretrained_encoder

logger = get_logger(__file__)

available_optimizers: Dict[str, Type[Optimizer]] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}

available_encoders: Dict[str, Type[nn.Module]] = {
    "vgg16": tv_models.vgg16,
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
    "resnet152": tv_models.resnet152,
    "alexnet": tv_models.alexnet,
    "densenet": tv_models.densenet161,
}


class Classifier(pl.LightningModule):
    @dataclass
    class HParams(Serializable):
        """ Hyperparameters of the LightningModule."""
        # Batch size. TODO: Do we need to change this when using dp or ddp?
        batch_size: int = 64

        # Number of hidden units (before the output head).
        # TODO: Should we use the hidden size of the pretrained encoder instead?
        # (At the moment, we create an additional Linear layer to map from the
        # hidden size of the pretrained encoder to our hidden_size.
        hidden_size: int = 2048

        # Which optimizer to use.
        optimizer: str = choice(available_optimizers.keys(), default="adam")
        # Prevent gradients of the classifier from backpropagating into the encoder.
        detach_classifier: bool = False
        # Learning rate of the optimizer.
        learning_rate: float = 0.001
        # L2 regularization term for the model weights.
        weight_decay: float = 1e-6
        # Use an encoder architecture from the torchvision.models package.
        encoder: str = choice(available_encoders.keys(), default="resnet18")
        # Retrain the encoder from scratch.
        train_from_scratch: bool = False

        # Determines which auxiliary task is used, and their corresponding args.
        aux_tasks: AuxiliaryTaskOptions = field(default_factory=AuxiliaryTaskOptions)

        # Wether to create one output head per task.
        # TODO: It makes no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = False

        def make_optimizer(self, *args, **kwargs) -> Optimizer:
            """ Creates the Optimizer object from the options. """
            global available_optimizers
            optimizer_class = available_optimizers[self.optimizer]
            options = {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
            options.update(kwargs)
            return optimizer_class(*args, **options)

    def __init__(self, hparams: HParams, config: Config):
        super().__init__()
        self.hp: "Classifier.HParams" = hparams
        self.config: Config = config
        self.data_module: LightningDataModule = self.config.make_datamodule()
        self.input_shape: Tuple[int, int, int] = self.data_module.dims
        self.classes = self.data_module.num_classes

        
        self.save_hyperparameters()
        # Metrics from the pytorch-lightning package.
        # TODO: Not sure how useful they really are or how to properly use them.
        self.acc = metrics.Accuracy()
        self.cm = metrics.ConfusionMatrix()

        self.classification_loss = nn.CrossEntropyLoss()
        encoder_model = available_encoders[self.hp.encoder]
        self.encoder, self.hp.hidden_size = get_pretrained_encoder(
            encoder_model=encoder_model,
            pretrained=not self.hp.train_from_scratch,
            freeze_pretrained_weights=False,
        )
        self.output = nn.Sequential(
            nn.Flatten(),  # type: ignore
            nn.Linear(self.hp.hidden_size, self.classes),
            nn.ReLU(),
        )
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.create_auxiliary_tasks()
        logger.debug(f"Log dir: {self.config.log_dir}")
        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))
            logger.debug("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                logger.debug(f"\t {task_name}: {task.coefficient}")


    def create_auxiliary_tasks(self) -> Dict[str, AuxiliaryTask]:
        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        AuxiliaryTask.hidden_size   = self.hp.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.output
        AuxiliaryTask.preprocessing = self.preprocess_inputs

        return self.hp.aux_tasks.create_tasks(
            input_shape=self.input_shape,
            hidden_size=self.hp.hidden_size
        )

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.encoder(x)
        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        y_pred = self.output(h_x)
        return y_pred

    def training_step(self, batch: Tuple[Tensor, Optional[Tensor]], batch_idx: int):
        self.train()
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx: int):
        self.eval()
        return self._shared_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx: int):
        self.eval()
        return self._shared_step(batch, batch_idx, prefix="test")

    def _shared_step(self, batch: Tuple[Tensor, Optional[Tensor]], batch_idx: int, prefix: str) -> Dict:
        x, y = batch
        loss_info = self.get_loss(x, y, name=prefix)
        # NOTE: loss is supposed to be a tensor, but I'm testing out giving a LossInfo object instead.
        return {
            "loss": loss_info.total_loss,
            "log": loss_info.to_log_dict(),
            "progress_bar": loss_info.to_pbar_message(),
            "loss_info": loss_info.detach(),
        }
    
    def get_loss(self, x: Tensor, y: Tensor=None, name: str="") -> LossInfo:
        # TODO: Add a clean input preprocessing setup.
        # TODO: [improvement] Support a mix of labeled / unlabeled data at the example-level.
        x, y = self.preprocess_inputs(x, y)
        h_x = self.encode(x)
        y_pred = self.output(h_x)

        total_loss = LossInfo(name=name, x=x, h_x=h_x, y_pred=y_pred, y=y)
        # Add the self-supervised losses from all the enabled auxiliary tasks.
        for task_name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task_loss = aux_task.get_scaled_loss(x, h_x=h_x, y_pred=y_pred, y=y)
                if self.config.verbose:
                    logger.debug(f"aux task {task_name}: LossInfo = {aux_task_loss}")
                total_loss += aux_task_loss

        if y is not None:
            supervised_loss = self.supervised_loss(y=y, y_pred=y_pred)
            total_loss += supervised_loss

        if self.config.debug and self.config.verbose:
            for name, loss in total_loss.losses.items():
                logger.debug(f"{name}, {loss.total_loss}, {loss.total_loss.requires_grad}, {loss.metrics}")
        return total_loss

    def supervised_loss(self, y: Tensor, y_pred: Tensor) -> LossInfo:
        loss = self.classification_loss(y_pred, y)
        loss_info = LossInfo(
            name=Tasks.SUPERVISED,
            total_loss=loss,
            y_pred=y_pred,
            y=y,
        )
        return loss_info

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        """ Customize the backward pass.
        Was thinking of using the LossInfo object as the loss, which is a bit hacky.
        """
        
        if isinstance(loss, LossInfo):
            loss.total_loss.backward()
        else:
            super().backward(trainer, loss, optimizer, optimizer_idx)

    def preprocess_inputs(self, x: Tensor, y: Tensor=None) -> Tuple[Dict[str, Tensor], Dict[str, Optional[Tensor]]]:
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
        # TODO: Re-add the multi-task stuff if needed.
        return x, y

    def logits(self, h_x: Tensor) -> Tensor:
        if self.hparams.detach_classifier:
            h_x = h_x.detach()
        return self.classifier(h_x)

    def encode(self, x: Tensor):
        x, _ = self.preprocess_inputs(x, None)
        return self.encoder(x)
    
    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
        ) -> Dict[str, Dict[str, Tensor]]:
        return self._shared_epoch_end(outputs)

    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
        ) -> Dict[str, Dict[str, Tensor]]:
        return self._shared_epoch_end(outputs)

    def _shared_epoch_end(
        self,
        outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]],
    ) -> Dict[str, Dict[str, Tensor]]:
        
        for output in outputs:
            loss_info = output["loss_info"]
            loss = output["loss"]
            assert loss.item() == loss_info.total_loss.item(), f"{loss} should be {loss_info.total_loss}"
        total_loss = sum(output["loss_info"] for output in outputs)
        return {
            "log": total_loss.to_log_dict(),
            "progress_bar": total_loss.to_pbar_message(),
        }

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    def prepare_data(self):
        # download
        self.data_module.prepare_data()

    def train_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.data_module.train_dataloader(
            batch_size=self.hp.batch_size,
        )
    
    def val_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.data_module.val_dataloader(
            batch_size=self.hp.batch_size,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.data_module.test_dataloader(
            batch_size=self.hp.batch_size,
        )
    
    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.hp.batch_size = value 
    
    @property
    def learning_rate(self) -> float:
        return self.hp.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.hp.learning_rate = value


class OldClassifier(nn.Module):
    @dataclass
    class HParams:
        """ Set of hyperparameters for the classifier.

        We use [simple_parsing](www.github.com/lebrice/simpleparsing) to
        generate command-line arguments for each attribute of this class.
        """
        batch_size: int = 128   # Input batch size for training.
        learning_rate: float = field(default=1e-3, alias="-lr")  # learning rate.

        # Dimensions of the hidden state (feature extractor/encoder output).
        hidden_size: int = 100

        # Hyperparameters of the "output head" module.
        output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

        # Use an encoder architecture from the torchvision.models package.
        encoder_model: Optional[str] = choice({
            "vgg16": tv_models.vgg16,
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "resnet101": tv_models.resnet101,
            "resnet152": tv_models.resnet152,
            "alexnet": tv_models.alexnet,
            # "squeezenet": models.squeezenet1_0,  # Not supported yet (weird output shape)
            "densenet": tv_models.densenet161,
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

        # Wether to create one output head per task.
        # TODO: It makes no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = False


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
        
        # Classifier for the default task.
        self.default_output_head = OutputHead(
            input_size=self.hidden_size,
            output_size=self.num_classes,
            hparams=self.hparams.output_head,
        )
        # Dictionary that maps from task classes to output head to be used.
        # By default, contains a single output head that serves all classes.
        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()  # type: ignore 
        logger.info(f"output heads: {self.output_heads}")

        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size
        AuxiliaryTask.input_shape   = self.input_shape
        AuxiliaryTask.encoder       = self.encoder
        AuxiliaryTask.classifier    = self.default_output_head # TODO: Also update this class attribute when switching tasks. 
        AuxiliaryTask.preprocessing = self.preprocess_inputs
        
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.hparams.aux_tasks.create_tasks(
            input_shape=input_shape,
            hidden_size=self.hparams.hidden_size
        )

        if self.config.debug and self.config.verbose:
            logger.debug(self)
            logger.debug("Auxiliary tasks:")
            for task_name, task in self.tasks.items():
                logger.debug(f"{task_name}: {task.coefficient}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.to(self.config.device)


    def supervised_loss(self, x: Tensor,
                              y: Tensor,
                              h_x: Tensor=None,
                              y_pred: Tensor=None,
                              loss_f: Callable[[Callable, Tensor],Tensor]=None) -> LossInfo:
        loss = self.classification_loss(y_pred, y)
        loss_info = LossInfo(
            name=Tasks.SUPERVISED,
            total_loss=loss,
            tensors=dict(x=x, h_x=h_x, y_pred=y_pred, y=y),
        )
        h_x = self.encode(x) if h_x is None else h_x
        y_pred = self.logits(h_x) if y_pred is None else y_pred
        y = y.view(-1)

        if loss_f is None:
            metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        else:
            loss = loss_f(self.classification_loss,y_pred)
        
        
        supervised_metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        
        return loss_info

    def preprocess_inputs(self, x: Tensor, y: Tensor=None) -> Tuple[Dict[str, Tensor], Dict[str, Optional[Tensor]]]:
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

        if y is not None and self.hparams.multihead:
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
        # Setting the current_task attribute also creates the output head if needed.
        self.current_task = task
        # also inform the auxiliary tasks that the task switched.
        for name, aux_task in self.tasks.items():
            if aux_task.enabled:
                aux_task.on_task_switch(task, **kwargs)

    def get_output_head(self, task: Task) -> OutputHead:
        """ Returns the output head for a given task.
        NOTE: Assumes that the model is multiheaded.
        """
        return self.output_heads[task.dumps()]

    @property
    def classifier(self) -> OutputHead:
        if self.hparams.multihead:
            return self.get_output_head(self.current_task)
        # Return the default output head.
        return self.default_output_head

    @property
    def current_task(self) -> Task:
        return self._current_task

    @current_task.setter
    def current_task(self, task: Task):
        """ Sets the current task.
        
        Used to create output heads when using a multihead model.
        """
        assert isinstance(task, Task), f"Please set the current_task by passing a `Task` object."
        self._current_task = task
        
        if not self.hparams.multihead:
            # not a multihead model, so we just return.
            logger.debug(f"just returning, since we're not a multihead model.")
            return

        task_str = task.dumps()
        if task_str not in self.output_heads:
            # If there isn't an output head for this task
            logger.debug(f"Creating a new output head for task {task}.")
            new_output_head = OutputHead(
                input_size=self.hidden_size,
                output_size=len(task.classes),
                hparams=self.hparams.output_head,
            ).to(self.device)

            # Store this new head in the module dict and add params to optimizer.
            self.output_heads[task_str] = new_output_head
            self.optimizer.add_param_group({"params": new_output_head.parameters()})

            task_head = new_output_head
        else:
            task_head = self.get_output_head(task)

        # Update the classifier used by auxiliary tasks:
        AuxiliaryTask.classifier = task_head

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
            task.on_model_changed(global_step=global_step, **kwargs)
