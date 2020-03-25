import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from simple_parsing import field, mutable_field
from simple_parsing.helpers import JsonSerializable
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import cuda_available, gpus_available


@dataclass
class Config:
    """Settings related to the training setup. """

    debug: bool = field(alias="-d", default=False, action="store_true", nargs=0)      # enable debug mode.
    verbose: bool = field(alias="-v", default=False, action="store_true", nargs=0)    # enable verbose mode.

    # Number of steps to perform instead of complete epochs when debugging
    debug_steps: Optional[int] = None
    data_dir: Path = Path("data")  # data directory.

    log_dir_root: Path = Path("results") # Logging directory.
    log_interval: int = 10   # How many batches to wait between logging calls.
    
    random_seed: int = 1            # Random seed.
    use_cuda: bool = cuda_available # Whether or not to use CUDA.
    
    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda")
    device: torch.device = torch.device("cuda" if cuda_available else "cpu")
    
    use_wandb: bool = True # Whether or not to log results to wandb
    # Name used to easily group runs together.
    # Used to create a parent folder that will contain the `run_name` directory. 
    run_group: Optional[str] = None 
    run_name: Optional[str] = None  # Wandb run name. If None, will use wandb's automatic name generation

    def __post_init__(self):
        # set the manual seed (for reproducibility)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.use_cuda and not cuda_available:
            print("Cannot use the passed value of argument 'use_cuda', as CUDA "
                  "is not available!")
            self.use_cuda = False
        if not self.use_cuda:
            self.device = torch.device("cpu")

        if self.debug:
            self.use_wandb = False
            self.run_name = "debug"
            if self.log_dir.exists():
                # wipe out the debug folder every time.
                shutil.rmtree(self.log_dir)
            
            self.log_dir.mkdir(exist_ok=False, parents=True)

            if self.use_cuda:
                # TODO: set CUDA deterministic.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    
    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(self.run_group or "", self.run_name or 'default')


# shared config object.
## TODO: unused, but might be useful!
config: Config = Config()
