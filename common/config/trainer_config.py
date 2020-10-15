""" Dataclass that holds the options (command-line args) for the Trainer
"""
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase

from simple_parsing import choice, field, mutable_field
from utils.serialization import Serializable

from .wandb_config import WandbLoggerConfig
from pathlib import Path
import os


@dataclass
class TrainerConfig(Serializable):
    """ Configuration options for the pytorch-lightning Trainer.
    
    TODO: Pytorch Lightning already has a mechanism for adding argparse
    arguments for the Trainer.. Would there be a better way of merging the
    simple-parsing and pytorch-lightning approaches ?
    """
    gpus: int = torch.cuda.device_count()
    overfit_batches: float = 0.
    fast_dev_run: bool = field(default=False, nargs=0, action="store_true")
    # Maximum number of epochs to train for.
    max_epochs: int = 10
    # Number of nodes to use.
    num_nodes: int = 1
    distributed_backend: str = "dp"
    log_gpu_memory: bool = False
    val_check_interval: Union[int, float] = 1.0
    auto_scale_batch_size: Optional[str] = None
    auto_lr_find: bool = False
    # Floating point precision to use in the model. (See pl.Trainer)
    precision: int = choice(16, 32, default=32)
    default_root_dir: Path = Path(os.getcwd()) / "results"

    # How much of training dataset to check (floats = percent, int = num_batches)
    limit_train_batches: Union[int, float] = 1.0
    # How much of validation dataset to check (floats = percent, int = num_batches)
    limit_val_batches: Union[int, float] = 1.0
    # How much of test dataset to check (floats = percent, int = num_batches)
    limit_test_batches: Union[int, float] = 1.0
    
    # Options for wandb logging.
    wandb: WandbLoggerConfig = mutable_field(WandbLoggerConfig)

    def create_loggers(self) -> Optional[Union[LightningLoggerBase, List[LightningLoggerBase]]]:
        if self.fast_dev_run:
            return None
        elif "pytest" in sys.modules:
            # Running inside a pytest session, not logging to wandb.
            return None
        return self.wandb.make_logger(wandb_parent_dir=self.log_dir_root)

    # TODO: These two aren't really used at the moment.

    # Root of where to store the logs.
    log_dir_root: Path = Path("results")
    
    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.wandb.project or ""),
            (self.wandb.group or ""),
            (self.wandb.run_name or ""),
            self.wandb.run_id,
        )
    
    
    def make_trainer(self,
                     callbacks: Optional[List[Callback]] = None) -> Trainer:
        """ Create a Trainer object from the command-line args.
        Adds the given loggers and callbacks as well.
        """
        loggers = self.create_loggers()
        return Trainer(
            logger=loggers,
            callbacks=callbacks,
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            max_epochs=self.max_epochs,
            distributed_backend=self.distributed_backend,
            log_gpu_memory=self.log_gpu_memory,
            overfit_batches=self.overfit_batches,
            fast_dev_run=self.fast_dev_run,
            auto_scale_batch_size=self.auto_scale_batch_size,
            auto_lr_find=self.auto_lr_find,
            # TODO: Either move the log-dir-related stuff from Config to this
            # class, or figure out a way to pass the value from Config to this
            # function
            default_root_dir=self.default_root_dir,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_train_batches,
        )
