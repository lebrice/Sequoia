""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from datasets import DatasetConfig
from pl_bolts.datamodules import LightningDataModule
from simple_parsing import (Serializable, choice, field, list_field,
                            mutable_field)

from .trainer_config import TrainerConfig
from .wandb_config import WandbLoggerConfig
from datasets.data_utils import FixChannels, keep_in_memory, train_valid_split
from pl_bolts.datamodules import LightningDataModule
from enum import Enum
from torchvision import transforms as transform_lib


@dataclass
class Config(Serializable):
    """ Options related to the experimental setup. """
    # Which dataset to use.
    # TODO: Eventually switch this to the type of 'Environment' data module to use.
    # TODO: Allow the customization of which transform to use depending on the command-line maybe
    # instead of having just a choice of which DatasetConfig object to use.
    dataset: DatasetConfig = mutable_field(DatasetConfig)
    
    log_dir_root: Path = Path("results")
    data_dir: Path = Path("data")
    # Run in Debug mode: no wandb logging, extra output.
    debug: bool = False
    # Enables more verbose logging.
    verbose: bool = False
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
        if self.debug:
            logger = None
        else:
            logger = self.wandb.make_logger(self.log_dir_root)
        return self.trainer.make_trainer(loggers=logger)

    def make_datamodule(self) -> LightningDataModule:
        """Creates the datamodule depending on the value of 'dataset' attribute.
        """
        return self.dataset.load(data_dir=self.data_dir)
