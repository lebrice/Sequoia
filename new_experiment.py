import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Type, Union, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from torchvision.datasets import MNIST

from config.pl_config import TrainerConfig, WandbLoggerConfig
from datasets import DatasetConfig, Datasets
from models.pretrained_model import get_pretrained_encoder
from simple_parsing import ArgumentParser, Serializable, choice, mutable_field, field
from utils.logging_utils import get_logger

logger = get_logger(__file__)


@dataclass
class HParams(Serializable):
    """ Model/Experiment Hyperparameters. """
    # Batch size. TODO: Do we need to change this when using dp or ddp?
    batch_size: int = 64

    # Number of hidden units (before the output head).
    # TODO: Should we use the hidden size of the pretrained encoder instead?
    # (At the moment, we create an additional Linear layer to map from the
    # hidden size of the pretrained encoder to our hidden_size.
    hidden_size: int = 2048

    # Which optimizer to use.
    optimizer: Type[Optimizer] = choice({
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }, default="adam")

    # Learning rate of the optimizer.
    learning_rate: float = 0.001
    # L2 regularization term for the model weights.
    weight_decay: float = 1e-6
    
    # Use an encoder architecture from the torchvision.models package.
    encoder_model: Type[nn.Module] = choice({
        "vgg16": tv_models.vgg16,
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
        "resnet101": tv_models.resnet101,
        "resnet152": tv_models.resnet152,
        "alexnet": tv_models.alexnet,
        "densenet": tv_models.densenet161,
    }, default="resnet18")
    # Use the pretrained weights of the ImageNet model from torchvision.
    pretrained_model: bool = False

    def make_optimizer(self, *args, **kwargs) -> Optimizer:
        options = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        options.update(kwargs)
        return self.optimizer(*args, **options)


@dataclass
class ExperimentConfig(Serializable):
    """ Options related to the experimental setup. """
    # Which dataset to use.
    dataset: DatasetConfig = choice({
        d.name: d.value for d in Datasets
    }, default=Datasets.mnist.name)
    
    log_dir_root: Path = Path("results")
    data_dir: Path = Path("data")

    debug: bool = False
    
    # Number of workers for the dataloaders.
    num_workers: int = torch.get_num_threads()

    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Options for wandb logging.
    wandb: WandbLoggerConfig = mutable_field(WandbLoggerConfig)
    # Options for the trainer.
    trainer: TrainerConfig = mutable_field(TrainerConfig)
    
    def __post_init__(self):
        self.set_seed()
    
    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.wandb.project or ""),
            (self.wandb.group or ""),
            (self.wandb.run_name or ""),
            self.wandb.run_id,
        )

    def make_trainer(self) -> Trainer:
        logger = None if self.debug else self.wandb.make_logger(self.log_dir_root)
        trainer = self.trainer.make_trainer(logger=logger)
        return trainer

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class MnistModel(pl.LightningModule):
    def __init__(self, classes: int, hparams: HParams, config: ExperimentConfig):
        super().__init__()
        self.classes = classes
        self.hp : HParams = hparams
        # self.hparams: HParams = self.hp.to_dict()
        self.config = config
        self.acc = metrics.Accuracy()
        self.cm = metrics.ConfusionMatrix()
        self.save_hyperparameters()
        self.batch_size = self.hp.batch_size

        self.encoder = get_pretrained_encoder(
            hidden_size=self.hp.hidden_size,
            encoder_model=self.hp.encoder_model,
            pretrained=self.hp.pretrained_model,
            freeze_pretrained_weights=False,                
        )
        self.output = nn.Sequential(
            nn.Flatten(),  # type: ignore
            nn.Linear(self.hp.hidden_size, self.classes),
            nn.ReLU(),
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

    def forward(self, x):
        h_x = self.encoder(x)
        y_pred = self.output(h_x)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        logs = {
            "train_loss": loss,
            "train_acc": self.acc(y_pred, y),
        }
        return {"loss": loss, "log": logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        logs = {
            "val_loss": loss,
            "val_acc": self.acc(y_pred, y),
        }
        return {"loss": loss, "log": logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_pred = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        logs = {
            "test_loss": loss,
            "test_acc": self.acc(y_pred, y),
        }
        return {"loss": loss, "log": logs}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        logs: List[Dict[str, Tensor]] = [
            output["log"] for output in outputs
        ]
        test_acc_mean = sum(log["val_acc"] for log in logs)
        test_acc_mean /= len(outputs)
        tqdm_dict = {'val_acc': test_acc_mean.item()}

        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'val_acc': test_acc_mean.item()}
        }
        return results

    
    def test_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        logs: List[Dict[str, Tensor]] = [
            output["log"] for output in outputs
        ]
        test_acc_mean = sum(log["test_acc"] for log in logs)
        test_acc_mean /= len(outputs)
        tqdm_dict = {'test_acc': test_acc_mean.item()}

        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'test_acc': test_acc_mean.item()}
        }
        return results

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters(), lr=0.001)

    def prepare_data(self):
        # download
        self.config.dataset.load(data_dir=self.config.data_dir, download=True)

    def setup(self, stage):
        train, test = self.config.dataset.load(data_dir=self.config.data_dir, download=False)
        n_train = int(0.8 * len(train))
        n_valid = len(train) - n_train
        # train/val split
        train, valid = torch.utils.data.random_split(train, [n_train, n_valid])
        # assign to use in dataloaders
        self.train_dataset = train
        self.valid_dataset = valid
        self.test_dataset = test

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.train_dataset)
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.valid_dataset)
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.test_dataset)

    def _get_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hp.batch_size,
            num_workers=self.config.num_workers,
        )


@dataclass
class Experiment(Serializable):
    # HyperParameters of the model/experiment.
    hparams: HParams = mutable_field(HParams)
    # Configuration options for an experiment (log_dir, etc.)
    config: ExperimentConfig = mutable_field(ExperimentConfig)

    def launch(self):
        dataset: DatasetConfig = self.config.dataset
        model = MnistModel(classes=dataset.num_classes, hparams=self.hparams, config=self.config)
        # most basic trainer, uses good defaults
        trainer = self.config.make_trainer()
        trainer.fit(model)
        # trainer.test(model)
        # trainer.save_checkpoint("test")
        # MnistModel.load_from_checkpoint("test")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Experiment, "experiment")
    args = parser.parse_args()
    experiment: Experiment = args.experiment
    experiment.launch()
