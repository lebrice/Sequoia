""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from simple_parsing import Serializable, choice, field, list_field, mutable_field

from datasets import DatasetConfig, Datasets


@dataclass
class WandbLoggerConfig(Serializable):
    """ Configuration options for the wandb logger. """
    # Which user to use
    entity: str = "lebrice"
    # The name of the project to which this run will belong.
    project: str = "pl_testing" 
    # Name used to easily group runs together.
    # Used to create a parent folder that will contain the `run_name` directory.
    # A unique string shared by all runs in a given group
    group: Optional[str] = None
    # Wandb run name. If None, will use wandb's automatic name generation
    run_name: Optional[str] = None
    # Identifier unique to each individual wandb run. When given, will try to
    # resume the corresponding run, generates a new ID each time.
    # TODO: Could also use a hash of the hparams, like @nitarshan did.     
    run_id: str = field(default_factory=wandb.util.generate_id)
    
    # Tags to add to this run with wandb.
    tags: List[str] = list_field()
    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None
    # Run offline (data can be streamed later to wandb servers).
    offline: bool = False
    # Enables or explicitly disables anonymous logging.
    anonymous: bool = False
    # Sets the version, mainly used to resume a previous run.
    version: Optional[str] = None
    # Save checkpoints in wandb dir to upload on W&B servers.
    log_model: bool = False
    
    def make_logger(self, wandb_parent_dir: Path) -> WandbLogger:
        wandb_logger = WandbLogger(
            name=self.run_name,
            save_dir=str(wandb_parent_dir),
            offline=self.offline,
            id=self.run_id,
            anonymous=self.anonymous,
            version=self.version,
            project=self.project,
            tags=self.tags,
            log_model=self.log_model,
            entity=self.entity,
            group=self.group,
        )
        return wandb_logger


@dataclass
class TrainerConfig(Serializable):
    """ Configuration options for the pytorch-lightning Trainer. """
    
    gpus: int = torch.cuda.device_count()
    overfit_batches: float = 0.
    fast_dev_run: bool = False
    max_epochs: int = 10
    # Number of nodes to use.
    num_nodes: int = 1
    distributed_backend: str = "dp"
    log_gpu_memory: bool = False
    auto_scale_batch_size: Optional[str] = None

    def make_trainer(self, logger: WandbLogger=None) -> Trainer:
        return Trainer(
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            max_epochs=self.max_epochs,
            distributed_backend=self.distributed_backend,
            logger=logger,
            log_gpu_memory=self.log_gpu_memory,
            overfit_batches=self.overfit_batches,
            fast_dev_run=self.fast_dev_run,
            auto_scale_batch_size=self.auto_scale_batch_size,
        )


