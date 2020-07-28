import functools
from utils.logging_utils import get_logger
import os
import shutil
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import tqdm
import wandb
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from simple_parsing import field, list_field, mutable_field
from utils import cuda_available, gpus_available, set_seed
from utils.early_stopping import EarlyStoppingOptions
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from .wandb_config import WandbConfig

logger = get_logger(__file__)

@dataclass
class Config(WandbConfig):
    """Settings related to the training setup. """

    debug: bool = field(alias="-d", default=False, action="store_true", nargs=0)      # enable debug mode.
    verbose: bool = field(alias="-v", default=False, action="store_true", nargs=0)    # enable verbose mode.

    # Number of steps to perform instead of complete epochs when debugging
    debug_steps: Optional[int] = None
    data_dir: Path = Path("data")  # data directory.

    log_interval: int = 10   # How many batches to wait between logging calls.

    log_interval_test_epochs: int = 10 # How many epochs to wait between test iterations.
    
    random_seed: int = 1            # Random seed.
    use_cuda: bool = cuda_available # Whether or not to use CUDA.
    
    # num_workers for the dataloaders.
    num_workers: int = torch.get_num_threads()

    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda") For multiple GPUs pass only indicies.
    #device: Tuple[torch.device] = (torch.device("cuda" if cuda_available else "cpu"),)
    device: torch.device = torch.device("cuda:0" if cuda_available else "cpu")
    
    use_wandb: bool = True # Whether or not to log results to wandb
    
    # Save the command-line arguments that were used to create this run.
    argv: List[str] = field(init=False, default_factory=sys.argv.copy)

    early_stopping: EarlyStoppingOptions = mutable_field(EarlyStoppingOptions)

    use_accuracy_as_metric: bool = False

    # Remove the existing log_dir, if any. Useful when debugging, as we don't
    # always want to keep some intermediate checkpoints around. 
    delete_existing_log_dir: bool = False

    def __post_init__(self):
        # set the manual seed (for reproducibility)
        set_seed(self.random_seed + (self.run_number or 0))
        

        if not self.run_group:
            # the run group is by default the name of the experiment.
            self.run_group = type(self).__qualname__.split(".")[0]
        
        if self.use_cuda and not cuda_available:
            print("Cannot use the passed value of argument 'use_cuda', as CUDA "
                  "is not available!")
            self.use_cuda = False
            
        if not self.use_cuda:
            self.device = torch.device("cpu")

        if self.debug:
            self.use_wandb = False
            if self.run_name is None:
                self.run_name = "debug"
            
           
            if self.use_cuda:
                # TODO: set CUDA deterministic.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        if self.delete_existing_log_dir and self.log_dir.exists():
            # wipe out the debug folder every time.
            shutil.rmtree(self.log_dir)
            logger.warning(f"REMOVED THE LOG DIR {self.log_dir}")
            self.log_dir.mkdir(exist_ok=False, parents=True)

        if self.notes:
            with open(self.log_dir / "notes.txt", "w") as f:
                f.write(self.notes)

# shared config object.
## TODO: unused, but might be useful!
# config: Config = Config()
