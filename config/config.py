""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, List, Optional, Type, Union

import torch
import wandb
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from torchvision import transforms as transform_lib

from datasets.data_utils import FixChannels, keep_in_memory, train_valid_split
from pl_bolts.datamodules import LightningDataModule
from setups.base import ExperimentalSetting
from simple_parsing import (Serializable, choice, field, list_field,
                            mutable_field)

from .trainer_config import TrainerConfig
from .wandb_config import WandbLoggerConfig


@dataclass
class Config(Serializable):
    """ Options related to the setup of an experiment (log_dir, cuda, debug etc.) """
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
    # Options for the Trainer object.
    trainer: TrainerConfig = mutable_field(TrainerConfig)

    def __post_init__(self):
        seed_everything(self.seed)
    

    def create_callbacks(self) -> List[Callback]:
        from experiments.callbacks.vae_callback import SaveVaeSamplesCallback
        from experiments.callbacks.knn_callback import KnnCallback
        return [
            SaveVaeSamplesCallback(),
            KnnCallback(),
        ]

    def create_loggers(self) -> Optional[Union[LightningLoggerBase, List[LightningLoggerBase]]]:
        if self.debug:
            logger = None
        else:
            logger = self.wandb.make_logger(wandb_parent_dir=self.log_dir_root)
        return logger

    def create_trainer(self, callbacks: Optional[List[Callback]]=None,
                             loggers: Optional[List[LightningLoggerBase]]=None,
                             trainer_config: Optional[TrainerConfig]=None) -> Trainer:
        """Creates a Trainer object from pytorch-lightning.

        NOTE: At the moment, uses the KNN and VAE callbacks.
        To use different callbacks, overwrite this method and pass different
        callbacks to the `self.config.create_trainer(callbacks=<some_callbacks>)
        method.

        Args:
            callbacks (Optional[List[Callback]], optional): list of callbacks
                to use. Defaults to the result of `self.create_callbacks()`.
            loggers (Optional[List[LightningLoggerBase]], optional): list of
                loggers to use. Defaults to the result of
                `self.create_loggers()`.
            trainer_config (Optional[TrainerConfig], optional): dataclass
                containing the keyword arguments to the Trainer constructor.
                Defaults to the value of `self.trainer`.

        Returns:
            Trainer: the Trainer object.
        """
        callbacks = callbacks or self.create_callbacks()
        loggers = loggers or self.create_loggers()
        trainer_config = trainer_config or self.trainer
        return trainer_config.make_trainer(loggers=loggers, callbacks=callbacks)

    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.wandb.project or ""),
            (self.wandb.group or ""),
            (self.wandb.run_name or ""),
            self.wandb.run_id,
        )
