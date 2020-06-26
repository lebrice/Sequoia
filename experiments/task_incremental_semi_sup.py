import itertools
from utils.logging_utils import get_logger
from collections import OrderedDict, defaultdict
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
from datasets import DatasetConfig, Datasets

from datasets.data_utils import unbatch, unlabeled, get_semi_sampler, get_lab_unlab_idxs
from datasets.subset import ClassSubset, Subset
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
    @dataclass
    class Config(TaskIncremental.Config):
        #wether to apply simclr augment
        ratio_labelled: float = 0.2
        #semi setup: 0 - only current task's unsupervised data, 1 - all tasks' unsupervised samples
        label_incremental: bool = 0 
        #wether to apply simclr augmentation
        simclr_augment: bool = False
        #stationary unlabeled dataset
        dataset_unlabeled: DatasetConfig = choice({
                d.name: d.value for d in Datasets
            }, default=Datasets.mnist.name)
        #use full unlabaled dataset 
        use_full_unlabeled: bool = True

    @dataclass
    class State(TaskIncremental.State):
        epoch:int = 0  
        idx_lab_unlab: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)

    # Experiment Configuration.
    config: Config = mutable_field(Config)     # Overwrite the type from Experiment.
    # Experiment state.
    state: State = mutable_field(State, init=False)        # Overwrite the type from Experiment.

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)

        #self.train_samplers_unlabeled: List[SubsetRandomSampler] = []
        #self.valid_samplers_unlabeled: List[SubsetRandomSampler] = []
        #self.test_samplers_unlabeled: List[SubsetRandomSampler] = []

        self.train_datasets_unlabeled: List[ClassSubset] = []
        self.epoch_length: Optional[int] = None
        self.batch_idx: Optional[int] = 0
        self.current_lr: Optional[float] = self.hparams.learning_rate
        self.full_train_dataset_unlabelled = None

    def load_task_datasets(self, tasks: List[Task]) -> None:
            """Create the train, valid, test, as well as corresponding semi-samplers and cumulative valid & test datasets
            for each task.
            """
            # download the dataset. 
            train_dataset, valid_dataset, test_dataset = super().load_datasets()
            if self.config.use_full_unlabeled:
                self.full_train_dataset_unlabelled, _ = self.config.dataset_unlabeled.load(data_dir=self.config.data_dir)
            
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

                self.train_samplers.append(None)
                self.valid_samplers.append(None)
                self.test_samplers.append(None)
                #get labeled and unlabeled indicies             
                idx_lab_new, idx_unlab_new = get_lab_unlab_idxs(train.targets, p=self.config.ratio_labelled)
                indices_train_lab, indices_train_unlab = self.state.idx_lab_unlab.setdefault(str(task.classes), (idx_lab_new, idx_unlab_new))
                
                #idx_unlab = np.setdiff1d(idx, self.state.idx_lab.get(task.classes))
                #sampler_train, sampler_train_unlabeled = get_semi_sampler(train.targets, p=self.config.ratio_labelled)
                #sampler_valid, sampler_valid_unlabeled = get_semi_sampler(valid.targets, p=1.)
                #sampler_test, sampler_test_unlabeled = get_semi_sampler(test.targets, p=1.)

                self.train_datasets.append(Subset(train, indices_train_lab))
                self.train_datasets_unlabeled.append(Subset(train, indices_train_unlab))
                self.valid_datasets.append(valid)
                self.test_datasets.append(test)

                #self.train_samplers.append(sampler_train)
                #self.train_samplers_unlabeled.append(sampler_train_unlabeled)

                #self.valid_samplers.append(sampler_valid)
                #self.valid_samplers_unlabeled.append(sampler_valid_unlabeled)

                #elf.test_samplers.append(sampler_test)
                #self.test_samplers_unlabeled.append(sampler_test_unlabeled)

            # Use itertools.accumulate to do the summation of the datasets.
            self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
            self.test_cumul_dataset = list(accumulate(self.test_datasets))

    def train(self, train_dataloader: Union[Dataset, DataLoader, Iterable], *args, **kwargs) -> TrainValidLosses:    
        if self.config.label_incremental:
            if self.full_train_dataset_unlabelled is None:
                train_dataloader_unlabeled = self.get_dataloader(self.full_train_dataset)
            else:
                logger.info("Using full unlabeled set")
                full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.full_train_dataset,self.full_train_dataset_unlabelled])
                train_dataloader_unlabeled = self.get_dataloader(full_unlabeled_dataset)
        else:
            if self.full_train_dataset_unlabelled is None:
                train_dataloader_unlabeled = self.get_dataloader(self.train_datasets_unlabeled[self.state.i])
            else:
                logger.info("Using full unlabeled set")
                full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.train_datasets_unlabeled[self.state.i],self.full_train_dataset_unlabelled])
                train_dataloader_unlabeled = self.get_dataloader(full_unlabeled_dataset)

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
                target = list(target)+([None]*len(u))
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
