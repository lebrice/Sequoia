import itertools
from utils.logging_utils import get_logger
from collections import OrderedDict, defaultdict
from datasets.ss_dataset import get_semi_sampler
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from common.losses import (LossInfo, TrainValidLosses, get_supervised_accuracy,
                           get_supervised_metrics)
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from common.task import Task
from datasets import DatasetConfig
from datasets.data_utils import unbatch, unlabeled
from datasets.subset import ClassSubset
from models.output_head import OutputHead
from simple_parsing import choice, field, list_field, mutable_field, subparsers
from simple_parsing.helpers import Serializable

from torch.utils.data.sampler import SubsetRandomSampler
from tasks import Tasks
from utils import utils
from experiments import TaskIncremental
from utils.utils import n_consecutive, roundrobin

from .experiment import Experiment

logger = get_logger(__file__)


@dataclass
class TaskIncremental_Semi_Supervised(TaskIncremental):
    """Task incremental semi-supervised setting
    """
    @dataclass
    class Config(Experiment.Config):
        """ Configuration options for the TaskIncremental_Semi-Supervised experiment. """
        #wether to apply simclr augment
        ratio_labelled: float = 0.2
        #semi setup: 0 - only current task's unsupervised data, 1 - all tasks' unsupervised samples
        semi_setup_full: bool = 0

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)

        self.epoch: Optional[int] = None  
        self.epoch_length: Optional[int] = None
        self.batch_idx: Optional[int] = 0
        self.current_lr: Optional[float] = self.hparams.learning_rate

def load_task_datasets(self, tasks: List[Task]) -> None:
        """Create the train, valid, test, as well as corresponding semi-samplers and cumulative valid & test datasets
        for each task.
        """
        # download the dataset.
        train_dataset, valid_dataset, test_dataset = super().load_datasets()
        assert valid_dataset # We have a validation dataset.

        self.full_train_dataset = train_dataset
        self.full_valid_dataset = valid_dataset
        self.full_test_dataset  = test_dataset

        # Clear the datasets for each task.
        self.train_datasets.clear()
        self.valid_datasets.clear()
        self.test_datasets.clear()

        for i, task in enumerate(tasks):
            train = ClassSubset(train_dataset, task)
            valid = ClassSubset(valid_dataset, task)
            test  = ClassSubset(test_dataset, task)

            sampler_train, sampler_train_unlabelled = get_semi_sampler(train.targets, p=self.ratio_labelled)
            sampler_valid, sampler_valid_unlabelled = get_semi_sampler(valid.targets, p=1.)
            sampler_test, sampler_test_unlabelled = get_semi_sampler(test.targets, p=1.)

            self.train_datasets.append((train,sampler_train,sampler_train_unlabelled))
            self.valid_datasets.append((valid,sampler_valid, sampler_valid_unlabelled))
            self.test_datasets.append((test,sampler_test,sampler_test_unlabelled))

        # Use itertools.accumulate to do the summation of the datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        self.test_cumul_dataset = list(accumulate(self.test_datasets))



if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(TaskIncremental, dest="experiment")

    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    experiment.launch()
