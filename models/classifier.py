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
from setups.base import ExperimentalSetting

class Classifier(pl.LightningModule):
    """ Classifier model.
    TODO Re-add the 'multihead' stuff and finish re-adding TaskIncremental

    [Improvements]
    - Move the 'classification' to an auxiliary task of some sort maybe?
    """

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

    def __init__(self, setting: ExperimentalSetting, hparams: HParams, config: Config):
        super().__init__()
        self.hp: "Classifier.HParams" = hparams
        self.config: Config = config
        self.setting: LightningDataModule = setting
        self.input_shape: Tuple[int, int, int] = self.setting.dims
        self.classes = self.setting.num_classes
        logger.debug(f"setting: {self.setting}")
        logger.debug(f"Input shape: {self.input_shape}, classes: {self.classes}")

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
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            # Not using the 'task' information on the individual examples just yet.
            x, y, _ = batch

        loss_info = self.get_loss(x, y, name=prefix)
        # NOTE: loss is supposed to be a tensor, but I'm testing out giving a LossInfo object instead.
        return {
            "loss": loss_info.total_loss,
            "log": loss_info.to_log_dict(),
            "progress_bar": loss_info.to_pbar_message(),
            "loss_info": loss_info.detach(),
        }
    
    def get_loss(self, x: Tensor, y: Tensor=None, name: str="") -> LossInfo:
        """Returns a LossInfo object containing the total loss and metrics. 

        Args:
            x (Tensor): The input examples.
            y (Tensor, optional): The associated labels. Defaults to None.
            name (str, optional): Name to give to the resulting loss object. Defaults to "".

        Returns:
            LossInfo: An object containing everything needed for logging/progressbar/metrics/etc.
        """
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
            assert loss.item() == loss_info.total_loss.item(), (
                f"{loss} should be {loss_info.total_loss}"
            )
        total_loss = sum(output["loss_info"] for output in outputs)
        return {
            "log": total_loss.to_log_dict(),
            "progress_bar": total_loss.to_pbar_message(),
        }

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    def prepare_data(self):
        # download
        self.setting.prepare_data(data_dir=self.config.data_dir)

    def train_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.setting.train_dataloader(
            batch_size=self.hp.batch_size,
        )
    
    def val_dataloader(self, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.setting.val_dataloader(
            batch_size=self.hp.batch_size,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.setting.test_dataloader(
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

