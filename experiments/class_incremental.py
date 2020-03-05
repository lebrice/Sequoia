from dataclasses import dataclass
from typing import Iterable, Union, List

from simple_parsing import choice, field, subparsers
from torch.utils.data import DataLoader

from common.losses import LossInfo
from config import Config

from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier
from experiments.experiment import Experiment
from experiments.iid import IID

@dataclass 
class ClassIncrementalConfig:
    class_incremental: bool = False  # train in a class-incremental fashion.
    n_classes_per_task: int = 2      # Number of classes per task.
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = False


@dataclass
class ClassIncremental(IID):
    config: Config = Config(class_incremental=True)
    online: bool = False  # wether or not to perform a single epoch of training.

    def __post_init__(self):
        if self.online:
            self.hparams.epochs = 1
        super().__post_init__()
