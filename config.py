import functools
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import tqdm
import wandb
from simple_parsing import field, mutable_field, list_field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.json_utils import JsonSerializable
from utils import cuda_available, gpus_available, set_seed
from utils.early_stopping import EarlyStoppingOptions


import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logging.getLogger('simple_parsing').addHandler(logging.NullHandler())

logger = logging.getLogger(__file__)


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
    
    # num_workers for the dataloaders.
    num_workers: int = 0

    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda")
    device: torch.device = torch.device("cuda" if cuda_available else "cpu")
    
    use_wandb: bool = True # Whether or not to log results to wandb
    
    project_name: str = "SSCL_replay" # project name to use in wandb.
    # Name used to easily group runs together.
    # Used to create a parent folder that will contain the `run_name` directory. 
    run_group: Optional[str] = None
    run_name: Optional[str] = None  # Wandb run name. If None, will use wandb's automatic name generation
    # An run number is used to differentiate different iterations of the same experiment.
    # Runs with the same name can be later grouped with wandb to produce stderr plots.
    run_number: Optional[int] = None 
    
    # Identifier unique to each individual wandb run. When given, will try to
    # resume the corresponding run, generates a new ID each time. 
    run_id: Optional[str] = None

    tags: List[str] = list_field() # Tags to add to this run with wandb.
    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None
    # Save the command-line arguments that were used to create this run.
    argv: List[str] = field(init=False, default_factory=sys.argv.copy)

    early_stopping: EarlyStoppingOptions = mutable_field(EarlyStoppingOptions)

    use_accuracy_as_metric: bool = False

    # Path where the wandb files should be stored. If the 'WANDB_DIR'
    # environment variable is set, uses that value. Otherwise, defaults to
    # the value of "<log_dir_root>/wandb"
    wandb_path: Optional[Path] = Path(os.environ['WANDB_DIR']) if "WANDB_DIR" in os.environ else None

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
            
            # logging.getLogger().setLevel(logging.DEBUG)
            # if self.log_dir.exists():
            #     # wipe out the debug folder every time.
            #     shutil.rmtree(self.log_dir)
            #     if self.log_dir.exists():
            #         # wipe out the debug folder every time.
            #         shutil.rmtree(self.log_dir)
            #         print(f"REMOVED THE LOG DIR {self.log_dir}")
            
            # self.log_dir.mkdir(exist_ok=False, parents=True)

            if self.use_cuda:
                # TODO: set CUDA deterministic.
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.project_name or ""),
            (self.run_group or ""),
            (self.run_name or 'default'),
            (f"run_{self.run_number}" if self.run_number is not None else ""),
        )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """ TODO: figure out if we should add handlers, etc. """
        try:
            p = Path(name)
            if p.exists():
                name = str(p.absolute().relative_to(Path.cwd()).as_posix())
        except:
            pass
        logger = logging.getLogger(name)
        return logger

    def wandb_init(self):    
        if self.run_name is None:
            # TODO: Create a run name using the coefficients of the tasks, etc?
            # At the moment, if no run name is given, the 'random' name from wandb is used.
            pass
        logger.info(f"Using wandb. Experiment name: {self.run_name}")
        config_dict = self.to_dict()

        if self.wandb_path is None:
            self.wandb_path = self.log_dir_root / "wandb"
        self.wandb_path.mkdir(parents=True, mode=0o777, exist_ok=True)

        if self.run_id is None:
            # TODO: add *proper* wandb resuming, probaby by using @nitarshan 's md5 id cool idea. 
            self.run_id = wandb.util.generate_id()
            # self.run_id = "-".join([self.run_group, self.run_name, str(self.run_number or 0)])

        logger.info(f"Wandb run id: {self.run_id}")

        run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            id=self.run_id,
            group=self.run_group,
            config=config_dict,
            dir=str(self.wandb_path),
            notes=self.notes,
            reinit=True,
            tags=self.tags,
            resume="allow",
        )
        wandb.run.save()

        if self.run_name is None:
            self.run_name = wandb.run.name
        
        print(f"Using wandb. Group name: {self.run_group} run name: {self.run_name}, log_dir: {self.log_dir}")
        return run

# shared config object.
## TODO: unused, but might be useful!
# config: Config = Config()

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)  
