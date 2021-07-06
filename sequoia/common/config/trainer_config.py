""" Dataclass that holds the options (command-line args) for the Trainer
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from simple_parsing import choice, field

from sequoia.common.hparams import HyperParameters, uniform
from .config import Config


@dataclass
class TrainerConfig(HyperParameters):
    """ Configuration options for the pytorch-lightning Trainer.
    
    TODO: Pytorch Lightning already has a mechanism for adding argparse
    arguments for the Trainer.. Would there be a better way of merging the
    simple-parsing and pytorch-lightning approaches ?
    """

    gpus: int = torch.cuda.device_count()
    overfit_batches: float = 0.0
    fast_dev_run: bool = field(default=False, nargs=0, action="store_true")

    # Maximum number of epochs to train for.
    max_epochs: int = uniform(1, 100, default=10)

    # Number of nodes to use.
    num_nodes: int = 1
    accelerator: Optional[str] = "dp" if gpus != 0 else None
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

    # If ``True``, enable checkpointing.
    # It will configure a default ModelCheckpoint callback if there is no user-defined
    # ModelCheckpoint in the `callbacks`.
    checkpoint_callback: bool = True

    def make_trainer(
        self,
        config: Config,
        callbacks: Optional[List[Callback]] = None,
        loggers: Iterable[LightningLoggerBase] = None,
    ) -> Trainer:
        """ Create a Trainer object from the command-line args.
        Adds the given loggers and callbacks as well.
        """
        # FIXME: Trying to subclass the DataConnector to fix issues while iterating
        # over gym envs, that arise because of the _with_is_last() function from
        # lightning.
        from pytorch_lightning.trainer.connectors.data_connector import DataConnector
        import pytorch_lightning.trainer.trainer

        setattr(pytorch_lightning.trainer.trainer, "DataConnector", DataConnector)
        trainer = Trainer(
            logger=loggers,
            callbacks=callbacks,
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
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
            checkpoint_callback=self.checkpoint_callback,
        )
        return trainer
