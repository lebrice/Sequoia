""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, Type, Union

import torch
import wandb
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from simple_parsing import (Serializable, choice, field, flag, list_field,
                            mutable_field)
from utils.parseable import Parseable

from .trainer_config import TrainerConfig
from .wandb_config import WandbLoggerConfig


@dataclass
class Config(Serializable, Parseable):
    """ Configuration options for an experiment.

    Contains all the command-line arguments for things that aren't supposed
    to be hyperparameters, but still determine how and experiment takes
    place. For instance, things like wether or not CUDA is used, or where
    the log directory is, etc.

    Extend this class whenever you want to add some command-line arguments
    for your experiment.
    """
    data_dir: Path = Path("data")
    # Run in Debug mode: no wandb logging, extra output.
    debug: bool = flag(False)
    # Enables more verbose logging.
    verbose: bool = flag(False)
    # Number of workers for the dataloaders.
    num_workers: int = torch.get_num_threads()
    # Random seed.
    seed: Optional[int] = None
    # Which device to use. Defaults to 'cuda' if available.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        self.seed_everything()

    def seed_everything(self) -> None:
        if self.seed is not None:
            seed_everything(self.seed)
