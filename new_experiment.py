import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models, transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from datasets import DatasetConfig, Datasets
from models.pretrained_model import get_pretrained_encoder
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics
from simple_parsing import ArgumentParser, Serializable, choice, mutable_field


@dataclass
class HParams(Serializable):
    """ Model/Experiment Hyperparameters. """
    # Batch size. TODO: Do we need to change this when using dp or ddp?
    batch_size: int = 64

    # Number of hidden units (before the output head)
    hidden_size: int = 2048

    # Which optimizer to use.
    optimizer: Type[Optimizer] = choice({
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }, default="adam")
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
class Config(Serializable):
    """ Options related to the experimental setup. """
    project: str = "PL-testing"
    run_name: str = "testing"

    
    # Which dataset to use.
    dataset: DatasetConfig = choice({
        d.name: d.value for d in Datasets
    }, default=Datasets.mnist.value)
    
    log_dir: Path = Path("results")
    data_dir: Path = Path("data")

    overfit_batches: Union[int, float] = 0.
    debug: bool = False
    fast_dev_run: bool = False

    max_epochs: int = 10
    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = torch.get_num_threads()
    num_gpus: int = torch.cuda.device_count()
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
        if self.debug:
            return None
        logger = pl_loggers.WandbLogger(
            name=self.run_name,
            save_dir=str(self.log_dir),
            # offline=self.offline,
            # id=self.id,
            # anonymous=self.anonymous,
            # version=self.version,
            project=self.project,
            # tags=self.tags,
            # log_model=self.log_model,
            # experiment=experiment,
            # entity=self.entity,
            # group=self.group,
        )
        return logger


class MnistModel(pl.LightningModule):

    def __init__(self, classes: int, hparams: HParams, config: Config):
        super().__init__()
        self.classes = classes
        self.hp : HParams = hparams or HParams()
        # self.hparams: HParams = self.hp.to_dict()
        self.config = config
        self.acc = metrics.Accuracy()
        self.cm = metrics.ConfusionMatrix()

        self.save_hyperparameters()
        self.encoder = get_pretrained_encoder(
            hidden_size=self.hp.hidden_size,
            encoder_model=self.hp.encoder_model,
            pretrained=self.hp.pretrained_model,
            freeze_pretrained_weights=False,                
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hp.hidden_size, self.classes),
            nn.ReLU(),
        )

    def forward(self, x):
        h_x = self.encoder(x)
        y_pred = self.output(h_x)
        return y_pred

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
            batch_size=self.hp.batch_size * (self.config.num_gpus or 1),
            num_workers=self.config.num_workers,
        )


@dataclass
class Experiment(Serializable):
    # HyperParameters of the model/experiment.
    hparams: HParams = mutable_field(HParams)
    # Configuration options for an experiment (log_dir, etc.)
    config: Config = mutable_field(Config)

    
    def launch(self):
        dataset: DatasetConfig = self.config.dataset

        model = MnistModel(classes=dataset.num_classes, hparams=self.hparams, config=self.config)
        # most basic trainer, uses good defaults
        trainer = Trainer(
            gpus=self.config.num_gpus,
            num_nodes=self.config.num_nodes,
            max_epochs=self.config.max_epochs,
            distributed_backend=self.config.distributed_backend,
            logger=self.config.get_logger(),
            log_gpu_memory=True,
            overfit_batches=self.config.overfit_batches,
            fast_dev_run=self.config.fast_dev_run,
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
