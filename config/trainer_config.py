from dataclasses import dataclass
from typing import Iterable, Optional, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from simple_parsing import choice
from utils.json_utils import Serializable

from .wandb_config import WandbLoggerConfig


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
    auto_lr_find: bool = False
    precision: int = choice(16, 32, default=32)

    def make_trainer(self, loggers: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]=True) -> Trainer:
        """ Create a Trainer object from the command-line args.
        Adds the given logger as well.
        """
        return Trainer(
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            max_epochs=self.max_epochs,
            distributed_backend=self.distributed_backend,
            logger=loggers,
            log_gpu_memory=self.log_gpu_memory,
            overfit_batches=self.overfit_batches,
            fast_dev_run=self.fast_dev_run,
            auto_scale_batch_size=self.auto_scale_batch_size,
            auto_lr_find=self.auto_lr_find,
        )
