import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Type, Union, List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics
from simple_parsing import ArgumentParser, Serializable, choice, mutable_field
from torch import optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from ..datasets import DatasetConfig

@dataclass
class HParams(Serializable):
    """ Model/Experiment Hyperparameters. """
    # Batch size. TODO: Do we need to change this when using dp or ddp?
    batch_size: int = 64
    # Which optimizer to use.
    optimizer: Type[Optimizer] = choice({
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }, default="adam")
    learning_rate: float = 0.001
    # L2 regularization term for the model weights.
    weight_decay: float = 1e-6
    # Momentum term (used for some optimizers).
    momentum: float = 0.
    
    def make_optimizer(self, *args, **kwargs) -> optim.Optimizer:
        options = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
        }
        options.update(kwargs)
        return self.optimizer(*args, **options)


@dataclass
class Config(Serializable):
    run_name: str = "testing"
    log_dir: Path = Path("logs")
    data_dir: Path = Path("data")
    max_epochs: int = 10
    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = torch.get_num_threads()
    num_gpus: int = -1
    num_nodes: int = 1
    distributed_backend: str = "dp"

    def __post_init__(self):
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
    
    def get_logger(self) -> pl_loggers.LightningLoggerBase:
        self.log_dir.mkdir(exist_ok=True, parents=True)
        print(f"creating a logger for log_dir {self.log_dir}")
        logger = pl_loggers.WandbLogger(
            name=self.run_name,
            save_dir=str(self.log_dir),
        )
        return logger


class MnistModel(pl.LightningModule):

    def __init__(self, classes: int, hparams: HParams=None, config: Config=None):
        super().__init__()
        print(f"classes: {classes}")
        print(f"hparams: {hparams}")
        print(f"config: {config}")
        self.classes = classes
        self.hp : HParams = hparams or HParams()
        # self.hparams: HParams = self.hp.to_dict()
        self.config = config

        self.acc = metrics.Accuracy(num_classes=self.classes)
        self.cm = metrics.ConfusionMatrix()

        self.save_hyperparameters()
        print(self.hparams)
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, self.classes)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        logs = {
            "train_loss": loss,
            "accuracy": self.acc(y_pred, y),
        }
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        logs = {"val_loss": loss}
        return {"loss": loss, "log": logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        logs = {"test_loss": loss}
        return {"loss": loss, "log": logs}

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters(), lr=0.001)

    def prepare_data(self):
        # download
        MNIST(str(self.config.data_dir), train=True, download=True, transform=transforms.ToTensor())
        MNIST(str(self.config.data_dir), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        mnist_train = MNIST(str(self.config.data_dir), train=True, download=False, transform=transforms.ToTensor())
        mnist_test = MNIST(str(self.config.data_dir), train=False, download=False, transform=transforms.ToTensor())
        # train/val split
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])
        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.train_dataset)
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.val_dataset)
    
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
    config: Config = mutable_field(Config)

    def launch(self):
        model = MnistModel(classes=10, hparams=self.hparams, config=self.config)
        # most basic trainer, uses good defaults
        trainer = Trainer(
            gpus=self.config.num_gpus,
            num_nodes=self.config.num_nodes,
            max_epochs=self.config.max_epochs,
            distributed_backend=self.config.distributed_backend,
            logger=self.config.get_logger(),
        )
        trainer.fit(model)
        trainer.test()
        trainer.save_checkpoint("test")
        MnistModel.load_from_checkpoint("test")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Experiment, "experiment")
    args = parser.parse_args()
    experiment: Experiment = args.experiment
    experiment.launch()
