import itertools
from utils.logging_utils import get_logger 
from collections import OrderedDict, defaultdict
from datasets.ss_dataset import get_semi_sampler
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate, cycle 
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
from tasks.simclr.simclr_task import SimCLRTask
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

logger = get_logger(__file__)
@dataclass 
class TaskIncremental_Semi_Supervised(TaskIncremental):
    """Task incremental semi-supervised setting
    """
    class Config(TaskIncremental.Config):
        #wether to apply simclr augment
        ratio_labelled: float = 0.2
        #semi setup: 0 - only current task's unsupervised data, 1 - all tasks' unsupervised samples
        label_incremental: bool = 0 
        #wether to apply simclr augmentation
        simclr_augment: bool = False

    class State(TaskIncremental.State):
        epoch:int = 0

    # Experiment Configuration.
    config: Config = mutable_field(Config)     # Overwrite the type from Experiment.
    # Experiment state.
    state: State = mutable_field(State, init=False)        # Overwrite the type from Experiment.

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)

        self.train_samplers_unlabeled: List[SubsetRandomSampler] = []
        self.valid_samplers_unlabeled: List[SubsetRandomSampler] = []
        self.test_samplers_unlabeled: List[SubsetRandomSampler] = []
    

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

                sampler_train, sampler_train_unlabeled = get_semi_sampler(train.targets, p=self.config.ratio_labelled)
                sampler_valid, sampler_valid_unlabeled = get_semi_sampler(valid.targets, p=1.)
                sampler_test, sampler_test_unlabeled = get_semi_sampler(test.targets, p=1.)

                self.train_datasets.append(train)
                self.train_samplers.append(sampler_train)
                self.train_samplers_unlabeled.append(sampler_train_unlabeled)

                self.valid_datasets.append(valid)
                self.valid_samplers.append(sampler_valid)
                self.valid_samplers_unlabeled.append(sampler_valid_unlabeled)

                self.test_datasets.append(test)
                self.test_samplers.append(sampler_test)
                self.test_samplers_unlabeled.append(sampler_test_unlabeled)

            # Use itertools.accumulate to do the summation of the datasets.
            self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
            self.test_cumul_dataset = list(accumulate(self.test_datasets))

    def train(self, train_dataloader: Union[Dataset, DataLoader, Iterable], *args, **kwargs) -> TrainValidLosses:    
        if self.config.label_incremental:
            train_dataloader_unlabeled = self.get_dataloader(self.full_train_dataset)
        else:
            train_dataloader_unlabeled = self.get_dataloader(self.train_datasets[self.state.i], self.train_samplers_unlabeled[self.state.i])
        train_dataloader = zip(cycle(train_dataloader), train_dataloader_unlabeled)
        self.epoch_length = len(train_dataloader_unlabeled)
        return super().train(train_dataloader, *args, **kwargs, steps_per_epoch=len(train_dataloader_unlabeled))

    def train_epoch(self, epoch, *args, **kwargs):
        self.state.epoch = epoch
        self.batch_idx = 0
        return super().train_epoch(epoch, *args, **kwargs)

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.batch_idx +=1 
        self.model.train()                    
        for batch_sup, batch_unsup in dataloader:  
            data, target = self.preprocess(batch_sup)
            u, _ = self.preprocess(batch_unsup)
            #create mixed batch  
            data = torch.cat([data, u])
            if target is not None:
                target = target.tolist()+([None]*len(u))
            yield self.train_batch(data, target)
    
    def step(self, global_step:int, **kwargs):
        return super().step(global_step, epoch=self.state.epoch, epoch_length=self.epoch_length, update_number=self.batch_idx, **kwargs)

    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        data, target = super().preprocess(batch)
        if self.config.simclr_augment:      
            data, target = SimCLRTask.preprocess_simclr(data, target, device=self.model.device)        
        return data, target

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(TaskIncremental_Semi_Supervised, dest="experiment")

    args = parser.parse_args()
    experiment: TaskIncremental_Semi_Supervised = args.experiment
    experiment.launch()
