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
from models.classifier import Classifier


@dataclass
class ConfigBase(Serializable):
    """ Options related to the experimental setup. """
    # Which dataset to use.
    # TODO: Eventually switch this to the type of 'Environment' data module to use.
    # TODO: Rework DatasetConfig so that you can choose the train/val/test augments, etc.
    # TODO: Rework DatasetConfig so they give LightningDataModule instances instead of Datasets directly. 
    dataset: DatasetConfig = choice({
            e.name: e.value for e in Datasets
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
        return self.dataset.load(data_dir=self.data_dir)


@dataclass
class Experiment(Serializable):
    
    @dataclass
    class Config(ConfigBase):
        pass
    
    # HyperParameters of the model/experiment.
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    # Configuration options for an experiment (log_dir, etc.)
    config: Config = mutable_field(Config)

    def launch(self):
        model = Classifier(hparams=self.hparams, config=self.config)
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
