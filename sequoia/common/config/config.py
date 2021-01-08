""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, Type, Union

import torch
import wandb
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pyvirtualdisplay import Display
from simple_parsing import (Serializable, choice, field, flag, list_field,
                            mutable_field)

from sequoia.utils.parseable import Parseable
# from .trainer_config import TrainerConfig
# from .wandb_config import WandbLoggerConfig


@dataclass
class Config(Serializable, Parseable):
    """ Configuration options for an experiment.

    TODO: This should contain configuration options that are not specific to
    either the Setting or the Method, or common to both. For instance, the
    random seed, or the log directory, wether CUDA is to be used, etc.
    """
    
    # Directory containing the datasets.
    data_dir: Path = Path("data")    
    # Directory containing the results of an experiment.
    log_dir: Path = Path("results")
    
    # Run in Debug mode: no wandb logging, extra output.
    debug: bool = flag(False)
    # Wether to render the environment observations. Slows down training.
    render: bool = flag(False)
    
    # Enables more verbose logging.
    verbose: bool = flag(False)
    # Number of workers for the dataloaders.
    num_workers: Optional[int] = torch.get_num_threads()
    # Random seed.
    seed: Optional[int] = None
    # Which device to use. Defaults to 'cuda' if available.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        self.seed_everything()

        if not self.render:
            # If `--render` isn't set, then try to create a virtual display.
            # This has the same effect as running the script with xvfb-run 
            try:
                display = Display(visible=0, size=(1366, 768))
                display.start()
            except Exception as e:
                raise RuntimeError(
                    f"Rendering is disabled, but we were unable to start the "
                    f"virtual display! {e}\n"
                    f"Make sure that xvfb is installed on your machine if you "
                    f"want to prevent rendering the environment's observations."
                )

    def seed_everything(self) -> None:
        if self.seed is not None:
            seed_everything(self.seed)
