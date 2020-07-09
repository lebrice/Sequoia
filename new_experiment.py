import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from pl_bolts.models import LogisticRegression
from pl_bolts.models.self_supervised import CPCV2, SimCLR
from pl_bolts.models.self_supervised.simclr import (SimCLREvalDataTransform,
                                                    SimCLRTrainDataTransform)
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import metrics, seed_everything
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
from simple_parsing import (ArgumentParser, Serializable, choice, field,
                            mutable_field)
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
        "rmsprop": optim.RMSprop,
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
    # Retrain the encoder from scratch.
    not_pretrained: bool = False

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
    # TODO: Eventually switch this to the type of 'Environment' data module to use.
    # TODO: Rework DatasetConfig so that you can choose the train/val/test augments, etc.
    # TODO: Rework DatasetConfig so they give LightningDataModule instances instead of Datasets directly. 
    dataset: Type[LightningDataModule] = choice({
        "mnist": MNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "fashion-mnist": FashionMNISTDataModule,
        "imagenet": ImagenetDataModule,
    }, default="mnist")
    
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
        seed_everything(self.seed)
    
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

    def make_datamodule(self) -> LightningDataModule:
        """Creates the datamodule depending on the value of 'dataset' attribute.
        """
        return self.dataset(self.data_dir, num_workers=self.num_workers)
        

class SelfSupervisedClassifierModel(pl.LightningModule):
    def __init__(self, hparams: HParams, config: ExperimentConfig):
        super().__init__()
        self.hp : HParams = hparams
        self.config = config
        self.data_module: LightningDataModule = self.config.make_datamodule()
        
        self.save_hyperparameters()
        # Metrics from the pytorch-lightning package.
        # TODO: Not sure how useful they really are or how to properly use them.
        self.acc = metrics.Accuracy()
        self.cm = metrics.ConfusionMatrix()

        self.classes = self.data_module.num_classes
        self.encoder = get_pretrained_encoder(
            hidden_size=self.hp.hidden_size,
            encoder_model=self.hp.encoder_model,
            pretrained=not self.hp.not_pretrained,
            freeze_pretrained_weights=False,                
        )
        self.hp.hidden_size = 512
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
        if isinstance(h_x, list) and len(h_x) == 1:
            h_x = h_x[0]
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


@dataclass
class Experiment(Serializable):
    # HyperParameters of the model/experiment.
    hparams: HParams = mutable_field(HParams)
    # Configuration options for an experiment (log_dir, etc.)
    config: ExperimentConfig = mutable_field(ExperimentConfig)

    def launch(self):
        model = SelfSupervisedClassifierModel(hparams=self.hparams, config=self.config)
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
