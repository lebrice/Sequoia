from utils.logging_utils import get_logger
import os
import pprint
import sys
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm
import wandb
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from simple_parsing import ArgumentParser, mutable_field
from common.losses import LossInfo, TrainValidLosses
from common.metrics import get_metrics
from models.classifier import Classifier
from models.cl_classifier import ContinualClassifier
from simple_parsing import ArgumentParser, choice, field, subparsers
from tasks import AuxiliaryTask, Tasks

from .experiment import Experiment

logger = get_logger(__file__)
from pytorch_lightning import Trainer
import wandb
"""Simple CL experiments.

TODO: Add the other supported scenarios from continuum here, since that would
probably be pretty easy:
- New Instances
- New Classes
- New Instances & Classes
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Type, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase

from config.trainer_config import TrainerConfig
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from simple_parsing import ArgumentParser, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from common.losses import LossInfo
from common.metrics import ClassificationMetrics, Metrics
from .experiment import Experiment
from models.classifier import Classifier
from setups.iid_setting import IIDSetting
logger = get_logger(__file__)
from .class_incremental import ClassIncrementalMethod, SelfSupervisedMethod


@dataclass
class IIDExperimentResults(Serializable):
    """TODO: A class which represents the results expected of a ClassIncremental CL experiment. """
    hparams: Classifier.HParams
    test_loss: float
    test_accuracy: float

from .class_incremental import ClassIncrementalMethod

@dataclass
class IIDMethod(ClassIncrementalMethod):
    """A Method gets applied to a Setting to produce some ExperimentalResults.

    The IID method is an extension of the ClassIncrementalMethod, in the sense
    that it reuses all the logic form ClassIncrementalMethod, but always has
    only one task containing all the classes.
    """
    # HyperParameters of the LightningModule (Classifier).
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    # Options for the Trainer.
    trainer: TrainerConfig = mutable_field(TrainerConfig)

    def __post_init__(self):
        self.setting: IIDSetting
        self.config: Experiment.Config

    def apply(self, setting: IIDSetting, config: Experiment.Config) -> IIDExperimentResults:
        """ Applies this method to the particular experimental setting.    
        """
        if not isinstance(setting, IIDSetting):
            raise RuntimeError(
                f"Can only apply this method on a IID setting or "
                f"on a setting which inherits from IIDSetting! "
                f"(setting is of type {type(setting)})."
            )
        
        self.setting = setting
        self.config = config

        logger.debug(f"Setting: {self.setting}")
        logger.debug(f"Config: {self.config}")

        self.trainer = self.create_trainer()
        self.model = self.create_model()

        n_tasks = self.setting.nb_tasks
        assert n_tasks == 1, f"iid setting has only one task!"
        logger.info(f"Number of tasks: {n_tasks}")
        
        # Just a sanity check:
        assert self.model.setting is self.setting is setting

        self.trainer.fit(self.model)
        test_outputs: List[Dict] = self.trainer.test()
        results = self.create_results_from_outputs(test_outputs)
        print(f"test results: {results}")
        return results 

    def create_results_from_outputs(self, outputs: List[Dict]) -> IIDExperimentResults:
        """Creates a Results object from the outputs of `self.trainer.test()`.
        """
        assert len(outputs) == 1
        output = outputs[0]
        test_loss: LossInfo = output["loss_info"]
        print(f"Test accuracy: {test_loss.accuracy:.2%}")        
        print(f"Test loss: {test_loss.total_loss}")
        return IIDExperimentResults(
            hparams=self.hparams,
            test_loss=test_loss.total_loss,
            test_accuracy=test_loss.accuracy,
        )

    def create_model(self) -> ContinualClassifier:
        model = ContinualClassifier(setting=self.setting, hparams=self.hparams, config=self.config)
        return model


@dataclass
class IID(Experiment):
    """ IID Experiment. """
    # Experimental Setting.
    setting: IIDSetting = mutable_field(IIDSetting)
    # Experimental method.
    method: SelfSupervisedMethod = mutable_field(SelfSupervisedMethod)

    @dataclass
    class Config(Experiment.Config):
        """ Config of an IID Experiment.

        Could use this to add some more command-line arguments if needed.
        """

    # Configuration options for the Experiment.
    config: Config = mutable_field(Config)

    def run(self):
        """ Simple IID Experiment. """
        logger.info(f"Starting the IID experiment with log dir: {self.config.log_dir}")
        results = self.method.apply(setting=self.setting, config=self.config)
        results.save(self.config.log_dir / "results.json")    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    
    args = parser.parse_args()
    experiment: IID = args.experiment
    experiment.launch()
