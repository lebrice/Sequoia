import os
import pprint
import sys
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import (Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type,
                    Union)

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common.losses import LossInfo, TrainValidLosses
from common.metrics import ClassificationMetrics, Metrics, get_metrics
from config.trainer_config import TrainerConfig
from models.cl_classifier import ContinualClassifier
from models.classifier import Classifier
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from setups.iid_setting import IIDSetting
from simple_parsing import (ArgumentParser, choice, field, mutable_field,
                            subparsers)
from tasks import AuxiliaryTask, Tasks
from utils.json_utils import Serializable
from utils.logging_utils import get_logger

from .class_incremental import ClassIncremental as BaseExperiment
from .class_incremental import ClassIncrementalMethod as BaseMethod
from .class_incremental import Results as BaseResults
from .experiment import Experiment, Method

logger = get_logger(__file__)


@dataclass
class Results(BaseResults):
    """Results of an IID Experiment. """
    hparams: Classifier.HParams
    test_loss: LossInfo


@dataclass
class IIDMethod(BaseMethod, Method[IIDSetting]):
    """A Method gets applied to a Setting to produce some ExperimentalResults.

    The IID method is an extension of the ClassIncrementalMethod, in the sense
    that it reuses all the logic form ClassIncrementalMethod, but always has
    only one task containing all the classes.
    """
    # HyperParameters of the LightningModule (Classifier).
    hparams: ContinualClassifier.HParams = mutable_field(ContinualClassifier.HParams)

    def __post_init__(self):
        self.setting: IIDSetting
        self.config: IID.Config

    def apply(self, setting: IIDSetting) -> Results:
        """ Applies this method to the particular experimental setting.    
        """
        if not isinstance(setting, IIDSetting):
            raise RuntimeError(
                f"Can only apply this method on a IID setting or "
                f"on a setting which inherits from IIDSetting! "
                f"(setting is of type {type(setting)})."
            )
        n_tasks = setting.nb_tasks
        assert n_tasks == 1, f"iid setting has only one task!"
        
        self.setting = setting

        logger.debug(f"Setting: {self.setting}")
        logger.debug(f"Config: {self.config}")

        return super().apply(setting)


@dataclass
class IID(BaseExperiment):
    """ IID Experiment.
    
    Currently, the hierarchy is quite simple, and the IID experiment is
    basically the same as the ClassIncremental one, except that there is only
    one task with all the classes in it.
    """
    # Experimental Setting.
    setting: IIDSetting = mutable_field(IIDSetting)
    # Experimental method.
    method: IIDMethod = mutable_field(IIDMethod)

    @dataclass
    class Config(BaseExperiment.Config):
        """ Config of an IID Experiment.

        Could use this to add some more command-line arguments if needed.
        """

    # Configuration options for the Experiment.
    config: Config = mutable_field(Config)

    def launch(self):
        """ Simple IID Experiment. """
        logger.info(f"Starting the IID experiment with log dir: {self.config.log_dir}")
        self.method.configure(self.config)
        results = self.method.apply(setting=self.setting)
        save_results_path = self.config.log_dir / "results.json"
        results.save(save_results_path)
        logger.info(f"Saved results of experiment at path {save_results_path}")
        return results


if __name__ == "__main__":
    IID.main()
