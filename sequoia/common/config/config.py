""" Config dataclasses for use with pytorch lightning.

@author Fabrice Normandin (@lebrice)
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pyvirtualdisplay import Display
from simple_parsing import Serializable, flag

from sequoia.utils.logging_utils import get_logger
from sequoia.utils.parseable import Parseable

# from .trainer_config import TrainerConfig
logger = get_logger(__name__)


virtual_display = None


@dataclass
class Config(Serializable, Parseable):
    """Configuration options for an experiment.

    TODO: This should contain configuration options that are not specific to
    either the Setting or the Method, or common to both. For instance, the
    random seed, or the log directory, wether CUDA is to be used, etc.
    """

    # Directory containing the datasets.
    data_dir: Path = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("DATA_DIR", "data")))
    # Directory containing the results of an experiment.
    log_dir: Path = Path(os.environ.get("RESULTS_DIR", "results"))

    # Run in Debug mode: no wandb logging, extra output.
    debug: bool = flag(False)
    # Wether to render the environment observations. Slows down training.
    render: bool = flag(False)

    # Enables more verbose logging.
    verbose: bool = flag(False)
    # Number of workers for the dataloaders.
    num_workers: Optional[int] = None
    # Random seed.
    seed: Optional[int] = None
    # Which device to use. Defaults to 'cuda' if available.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.seed_everything()
        self._display: Optional[Display] = None
        self.rng = np.random.default_rng(self.seed)
        self.log_dir = Path(self.log_dir)
        self.data_dir = Path(self.data_dir)

    def __del__(self):
        if self._display:
            self._display.stop()

    def get_display(self) -> Optional[Display]:
        if self._display:
            return self._display
        if not self.render:
            # If `--render` isn't set, then try to create a virtual display.
            # This has the same effect as running the script with xvfb-run
            try:
                virtual_display = Display(visible=False, size=(1366, 768))
                virtual_display.start()
                self._display = virtual_display
            except Exception as e:
                logger.warning(
                    RuntimeWarning(
                        f"Rendering is disabled, but we were unable to start the "
                        f"virtual display! {e}\n"
                        f"Make sure that xvfb is installed on your machine if you "
                        f"want to prevent rendering the environment's observations."
                    )
                )
        return self._display

    def seed_everything(self) -> None:
        if self.seed is not None:
            seed_everything(self.seed)
