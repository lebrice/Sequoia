from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Type, Union

import matplotlib.pyplot as plt
import torch
import tqdm
import wandb
from simple_parsing import choice, field, subparsers, mutable_field
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.losses import LossInfo
from common.metrics import Metrics
from config import Config
from datasets import Dataset
from datasets.mnist import Mnist
from datasets.fashion_mnist import FashionMnist
from models.classifier import Classifier
from tasks import AuxiliaryTask
from utils import utils


@dataclass  # type: ignore
class Experiment:
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    # Model Hyper-parameters
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    
    dataset: Dataset = choice({
        "mnist": Mnist(),
        "fashion_mnist": FashionMnist(),
    }, default="mnist")

    config: Config = Config()
    model: Classifier = field(default=None, init=False)
    model_name: str = field(default=None, init=False)

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.
        Additionally, the fields created here are not added in the wandb logs.       
        """
        AuxiliaryTask.input_shape   = self.dataset.x_shape

        # Set these shared attributes so that all the Auxiliary tasks can be created.
        if isinstance(self.dataset, (Mnist, FashionMnist)):
            AuxiliaryTask.input_shape = self.dataset.x_shape
            AuxiliaryTask.hidden_size = self.hparams.hidden_size

        self.model = self.get_model(self.dataset).to(self.config.device)
        self.model_name = type(self.model).__name__

        self.train_loader: DataLoader = NotImplemented
        self.valid_loader: DataLoader = NotImplemented

        self.global_step: int = 0

    @abstractmethod
    def run(self):
        pass

    def load(self):
        """ Setup the dataloaders and other settings before training. """
        self.dataset.load(self.config)
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders
        self.global_step = 0

    def get_model(self, dataset: Dataset) -> Classifier:
        if isinstance(dataset, (Mnist, FashionMnist)):
            from models.mnist import MnistClassifier
            return MnistClassifier(
                hparams=self.hparams,
                config=self.config,
            )
        raise NotImplementedError("TODO: add other models for other datasets.")


    def log(self, message: Union[str, Dict, LossInfo], value: Any=None, step: int=None, once: bool=False, prefix: str="", always_print: bool=False):
        if always_print or (self.config.debug and self.config.verbose):
            print(message, value if value is not None else "")
        if self.config.use_wandb:
            # if we want to long once (like a final result, step should be None)
            # else, if not given, we use the global step.
            step = None if once else (step or self.global_step)
            
            message_dict: Dict = message
            if message is None:
                return
            if isinstance(message, dict):
                message_dict = message
            elif isinstance(message, LossInfo):
                message_dict = message.to_log_dict()
            elif isinstance(message, str) and value is not None:
                message_dict = {message: value}
            else:
                message_dict = message
            
            if prefix:
                message_dict = utils.add_prefix(message_dict, prefix)
            
            wandb.log(message_dict, step=step)

    def train_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        batch_size = data.shape[0]
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)

        total_loss = batch_loss_info.total_loss
        losses     = batch_loss_info.losses
        tensors    = batch_loss_info.tensors
        metrics    = batch_loss_info.metrics

        total_loss.backward()
        self.model.optimizer.step()
        return batch_loss_info

    def train_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()

        pbar = tqdm.tqdm(dataloader) # disable=not (self.config.verbose or self.config.debug)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.model.device)
            target = target.to(self.model.device)
            batch_size = data.shape[0]

            batch_loss_info = self.train_batch(batch_idx, data, target)
            yield batch_loss_info

            self.global_step += batch_size

            if batch_idx % self.config.log_interval == 0:
                pbar.set_description(f"Train Epoch {epoch}")
                message = self.pbar_message(batch_loss_info)
                pbar.set_postfix(message)

                self.log(batch_loss_info, prefix="Train ")


    def test_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        return self.model.get_loss(data, target)

    def test_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.eval()
        test_loss = 0.

        test_loss = LossInfo()

        pbar = tqdm.tqdm(dataloader)
        pbar.set_description(f"Test Epoch {epoch}")

        for i, (data, target) in enumerate(pbar):
            with torch.no_grad():
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                batch_loss = self.test_batch(i, data, target)
                yield batch_loss
                test_loss += batch_loss
       
        return test_loss

    def pbar_message(self, batch_loss_info: LossInfo) -> Dict:
        message: Dict[str, Any] = OrderedDict()
        # average_accuracy = (overall_loss_info.metrics.get("accuracy", 0) / (batch_idx + 1))
        message["Total Loss"] = batch_loss_info.total_loss.item()
        message["metrics"] =   batch_loss_info.metrics        
        # add the logs for all the scaled losses:
        
        for loss_name, loss_tensor in batch_loss_info.losses.items():
            if loss_name.endswith("_scaled"):
                continue
            
            scaled_loss_tensor = batch_loss_info.losses.get(f"{loss_name}_scaled")

            if scaled_loss_tensor is not None:
                message[loss_name] = f"{utils.loss_str(scaled_loss_tensor)} ({utils.loss_str(loss_tensor)})"
            else:
                message[loss_name] = utils.loss_str(loss_tensor)
        return message




    @property
    def plots_dir(self) -> Path:
        path = self.config.log_dir / "plots"
        if not path.is_dir():
            path.mkdir()
        return path

    @property
    def samples_dir(self) -> Path:
        path = self.config.log_dir / "samples"
        if not path.is_dir():
            path.mkdir()
        return path

    @property
    def checkpoints_dir(self) -> Path:
        path = self.config.log_dir / "checkpoints"
        if not path.is_dir():
            path.mkdir()
        return path
