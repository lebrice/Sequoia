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
try:
    from tasks.simclr.falr.config import HParams
    from tasks.simclr.falr.data import SimCLRAugment
    from tasks.simclr.falr.losses import SimCLRLoss
    from tasks.simclr.falr.models import Projector
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()

logger = get_logger(__file__)
@dataclass
class TaskIncremental_Semi_Supervised(TaskIncremental):
    """Task incremental semi-supervised setting
    """
    #wether to apply simclr augment
    ratio_labelled: float = 0.2
    #semi setup: 0 - only current task's unsupervised data, 1 - all tasks' unsupervised samples
    label_incremental: bool = 0
    #wether to apply simclr augmentation
    simclr_augment: bool = False

    #for mixup of labeled data: the alpha parameter for the beta distribution from where the mixing lambda is drawn (mainly implemented for ICT)
    mixup_sup_alpha: float = 0.
    #TODO: move it to aux-task

    #lr scheduling: length of learning rate rampup in the beginning (mainly for ICT)
    lr_rampup: int = 5
    #length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate reaches to zero (mainly for ICT)
    lr_rampdown_epochs: int = 0

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)

        self.train_samplers_unlabeled: List[SubsetRandomSampler] = []
        self.valid_samplers_unlabeled: List[SubsetRandomSampler] = []
        self.test_samplers_unlabeled: List[SubsetRandomSampler] = []
    

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

                sampler_train, sampler_train_unlabeled = get_semi_sampler(train.targets, p=self.ratio_labelled)
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
        if self.label_incremental:
            train_dataloader_unlabeled = self.get_dataloader(self.full_train_dataset)
        else:
            train_dataloader_unlabeled = self.get_dataloader(self.train_datasets[self.state.i], self.train_samplers_unlabeled[self.state.i])
        train_dataloader = zip(cycle(train_dataloader), train_dataloader_unlabeled)
        self.epoch_length = len(train_dataloader_unlabeled)
        return super().train(train_dataloader, *args, **kwargs, steps_per_epoch=len(train_dataloader_unlabeled))

    def train_epoch(self, epoch, *args, **kwargs):
        self.epoch = epoch
        self.batch_idx = 0
        return super().train_epoch(epoch, *args, **kwargs)

    def preprocess_simclr(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
            data = batch[0].to(self.model.device)
            target = batch[1].to(self.model.device) if len(batch) == 2 else None  # type: ignore

            self.options = SimCLRTask.Option
            # Set the same values for equivalent hyperparameters
            self.options.image_size = data.shape[-1]
            self.options.double_augmentation = False

            self.augment = Compose([
                ToPILImage(),
                SimCLRAugment(self.options),
                Lambda(lambda tup: torch.stack([tup[0], tup[1]]))
            ])

            data = torch.cat([self.augment(x_i) for x_i in data.cpu()], dim=0)  # [2*B, C, H, W]
            target = torch.cat([ torch.stack([t,t]) for t in target.cpu()], dim=0) if target is not None else None
            return data.to(self.config.device), target.to(self.config.device)
    
    def preprocess_sup_mixup(self, x,y):
        from tasks.mixup import mixup_data_sup
        def mixup_criterion(y_a, y_b, lam):
            return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        mixed_input, target_a, target_b, lam = mixup_data_sup(x, y, self.mixup_sup_alpha)
        mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
        loss_func = mixup_criterion(target_a_var, target_b_var, lam)
        return mixed_input_var, loss_func

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.batch_idx +=1 
        self.model.train()
        for batch_sup, batch_unsup in dataloader:   
            data, target = self.preprocess(batch_sup) if not self.simclr_augment else self.preprocess_simclr(batch_sup)
            u, _ = self.preprocess(batch_unsup) if not self.simclr_augment else self.preprocess_simclr(batch_unsup)
            yield self.train_batch_semi(data, target, u)

    def train_batch_semi(self, data: Tensor, target: Optional[Tensor], u: Tensor) -> LossInfo:
        if self.lr_rampdown_epochs > 0:
            self.current_lr = self.adjust_learning_rate(self.model.optimizer, self.epoch, self.batch_idx, self.epoch_length, self.supervised_epochs_per_task)
        self.log({'lr':self.current_lr})

        self.model.optimizer.zero_grad()
        loss_f = None                           
        data, target = self.model.preprocess_inputs(data, target)
        if self.mixup_sup_alpha>0:
            data, loss_f = self.preprocess_sup_mixup(data,target)

        batch_loss_info = self.model.get_loss(u, None)
        if target is not None:
            batch_loss_info = batch_loss_info + self.model.supervised_loss(data, target, loss_f=loss_f)

        total_loss = batch_loss_info.total_loss
        total_loss.backward()
        self.model.optimizer_step(global_step=self.global_step,
                                    epoch=self.epoch,           
                                    epoch_length=self.epoch_length,
                                    update_number=self.batch_idx)
        self.global_step += data.shape[0]
        return batch_loss_info




if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(TaskIncremental_Semi_Supervised, dest="experiment")

    args = parser.parse_args()
    experiment: TaskIncremental_Semi_Supervised = args.experiment
    experiment.launch()
