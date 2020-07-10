import os
import hashlib
import tqdm
import itertools
from utils.logging_utils import get_logger
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate, cycle 
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import Tensor

from models.classifier import Classifier
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image
from utils.early_stopping import EarlyStoppingOptions, early_stopping
from common.losses import (LossInfo, TrainValidLosses, get_supervised_accuracy,
                           get_supervised_metrics, AUC_Meter)   
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from common.task import Task
from datasets import DatasetConfig, Datasets

from datasets.data_utils import unbatch, unlabeled, get_semi_sampler, get_lab_unlab_idxs, train_valid_split
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
        use_full_unlabeled: bool = False


        reduce_full_unlabeled: float = 0.2
        datasets_dir: Path = Path(os.environ["HOME"]).joinpath('data')

    @dataclass
    class State(TaskIncremental.State):
        epoch:int = 0  
        idx_lab_unlab: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)
        perf_meter: AUC_Meter = mutable_field(AUC_Meter, repr=False, init=False)


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

        print(f'\n {torch.get_num_threads()} cpu cores available \n')

    def get_preprocessed_dataset_path(self, dataset_name: str, unique_identifiers:List, rootpath:Path)->Path:
                #model is identifies by number of epochs for pretraining, encoder type and aux_tasks coefficients
                name = hashlib.md5(str([dataset_name]+unique_identifiers).encode('utf-8')).hexdigest()
                path = rootpath.joinpath('sscl_datasets')
                path.mkdir(parents=True, exist_ok=True)
                path = path.joinpath(f'{dataset_name}_{name}') 
                return path
    
    def load_preprocessed_dataset(self, path: Path):
                try:
                    X = torch.from_numpy(np.load(str(path)+'data.npy', allow_pickle=True))
                    Y = torch.from_numpy(np.load(str(path)+'label.npy', allow_pickle=True))
                    logger.info(f"Loaded preprocesses dataset from {path}")
                    return TensorDataset(X,Y)

                except FileNotFoundError:
                    return None
    
    def save_preprocessed_dataset(self, dataset_, path: Path):
                #convert to TensorDataset
                X = []
                Y = []
                logger.info(f"Saving preprocessed dataset to {path}")
                for (x,y) in tqdm.tqdm(dataset_, total=len(dataset_)):
                    X.append(x)
                    Y.append(y)
                X = torch.stack(X).numpy()
                Y = torch.tensor(Y).numpy()
                np.save(str(path)+'data.npy', X, allow_pickle=True)
                np.save(str(path)+'label.npy', Y, allow_pickle=True)
                

    def load_task_datasets(self, tasks: List[Task]) -> None:
            """Create the train, valid, test, as well as corresponding semi-samplers and cumulative valid & test datasets
            for each task.
            """
            
            # download the dataset. 
            super().load_task_datasets(tasks)
            train_dataset, valid_dataset, test_dataset = super().load_datasets()

            if self.config.use_full_unlabeled:
                    path_to_load = self.get_preprocessed_dataset_path(self.config.dataset_unlabeled.name, [self.config.dataset_unlabeled.num_classes, self.config.dataset_unlabeled.x_shape, self.config.reduce_full_unlabeled], rootpath=self.config.data_dir)
                    #try to load preprocessed dataset
                    full_train_dataset_unlabelled = self.load_preprocessed_dataset(path_to_load)
                    
                    #create if coulnt load it
                    if full_train_dataset_unlabelled is None:
                        full_train_dataset_unlabelled, _ = self.config.dataset_unlabeled.load(data_dir=self.config.data_dir)

                        if self.config.reduce_full_unlabeled > 0:
                            idx_full_unlab, _ = get_lab_unlab_idxs(full_train_dataset_unlabelled.targets, p=self.config.reduce_full_unlabeled)
                            full_train_dataset_unlabelled = Subset(full_train_dataset_unlabelled, idx_full_unlab)
                        path_to_save = self.get_preprocessed_dataset_path(self.config.dataset_unlabeled.name, [self.config.dataset_unlabeled.num_classes, self.config.dataset_unlabeled.x_shape, self.config.reduce_full_unlabeled], rootpath=self.config.datasets_dir)
                        self.save_preprocessed_dataset(full_train_dataset_unlabelled, path_to_save)

                    self.full_train_dataset_unlabelled = full_train_dataset_unlabelled


            
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

    def train(self,
              train_dataloader: Union[Dataset, DataLoader, Iterator],                
              valid_dataloader: Union[Dataset, DataLoader, Iterator],    
              test_dataloader:Union[Dataset, DataLoader, Iterator],
              epochs: int,                
              description: str=None,
              early_stopping_options: EarlyStoppingOptions=None,
              use_accuracy_as_metric: bool=None,                
              temp_save_dir: Path=None,
              steps_per_epoch: int = None) -> TrainValidLosses:
        if not 'pretrain' in description: 
            #half of the batch comes from labeled and half from unlabeled data
            batch_size = int(self.hparams.batch_size / 2)
            #labeled loader with a new batch size
            if isinstance(train_dataloader, DataLoader):
                train_dataloader = self.get_dataloader(train_dataloader.dataset, batch_size=batch_size)
            else:
                train_dataloader = self.get_dataloader(train_dataloader, batch_size=batch_size)
            
            if self.config.label_incremental:
                if self.full_train_dataset_unlabelled is None:
                    train_dataloader_unlabeled = self.get_dataloader(self.full_train_dataset, batch_size=batch_size)
                else:
                    logger.info("Using full unlabeled set")
                    full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.full_train_dataset,self.full_train_dataset_unlabelled])
                    train_dataloader_unlabeled = self.get_dataloader(full_unlabeled_dataset, batch_size=batch_size)
            else:
                if self.full_train_dataset_unlabelled is None:
                    train_dataloader_unlabeled = self.get_dataloader(self.train_datasets_unlabeled[self.state.i], batch_size=batch_size)
                else:
                    logger.info("Using full unlabeled set")
                    full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.train_datasets_unlabeled[self.state.i],self.full_train_dataset_unlabelled])
                    train_dataloader_unlabeled = self.get_dataloader(full_unlabeled_dataset, batch_size=batch_size)

            train_dataloader = zip(cycle(train_dataloader), train_dataloader_unlabeled)
            self.epoch_length = len(train_dataloader_unlabeled)
            steps_per_epoch = len(train_dataloader_unlabeled)

        return super().train(train_dataloader, valid_dataloader, test_dataloader, 
                            epochs, description, early_stopping_options, use_accuracy_as_metric,
                            temp_save_dir, steps_per_epoch)

    def train_epoch(self, epoch, *args, **kwargs):
        self.state.epoch = epoch
        self.batch_idx = 0 
        return super().train_epoch(epoch, *args, **kwargs)

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.batch_idx +=1 
        self.model.train()                    
        for batch in dataloader: 
            if len(batch)==2:
                batch_sup, batch_unsup = batch
                data, target = self.preprocess(batch_sup)
                u, _ = self.preprocess(batch_unsup)
                #create mixed batch  
                data = torch.cat([data, u])
                if target is not None:
                    target = torch.stack(list(target)+([torch.tensor(-1).to(self.model.out_device)]*len(u)))
            else:
                data, target = self.preprocess(batch)
            yield self.train_batch(data, target)
        
    def train_batch(self, *args, **kwargs) -> LossInfo:
        loss = super().train_batch(*args, **kwargs)
        loss = self.state.perf_meter.update(loss)
        return loss

    def step(self, global_step:int, **kwargs): 
        return super().step(global_step, epoch=self.state.epoch, epoch_length=self.epoch_length, update_number=self.batch_idx, **kwargs)

    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        data, target = super().preprocess(batch)
        if self.config.simclr_augment: 
            data, target = SimCLRTask.preprocess_simclr(data, target)
        return data, target

    def test(self, dataloader: Union[Dataset, DataLoader], description: str = None, name: str = "Test") -> LossInfo:
        loss_info = super().test(dataloader, description, name)
        loss_info = self.state.perf_meter.update(loss_info).detach()
        #log validation EPOCH performance
        if loss_info.name=='Valid_full':
            self.log({'Valid_full':loss_info.to_dict()})
        return loss_info
    
    def test_batch(self, *args, **kwargs):
        loss = super().test_batch(*args, **kwargs)
        loss = self.state.perf_meter.update(loss)
        return loss
    
    def load_pretrain_dataset(self):
        if self.config.unsupervised_epochs_pretraining>0:
                    path = self.get_preprocessed_dataset_path(self.config.pretraining_dataset.name, [self.config.pretraining_dataset.num_classes, self.config.pretraining_dataset.x_shape, self.config.reduce_full_unlabeled])
                    #try to load preprocessed dataset
                    pretrain_train_dataset = self.load_preprocessed_dataset(path)
                    
                    #create if coulnt load it
                    if pretrain_train_dataset is None:
                        pretrain_train_dataset, _ = self.config.pretraining_dataset.load(data_dir=self.config.data_dir)

                        if self.config.reduce_full_unlabeled > 0:
                            idx_pretrain_unlab, _ = get_lab_unlab_idxs(pretrain_train_dataset.targets, p=self.config.reduce_full_unlabeled)
                            pretrain_train_dataset = Subset(pretrain_train_dataset, idx_pretrain_unlab)
                        self.save_preprocessed_dataset(pretrain_train_dataset, path)

                    self.pretrain_train_dataset, self.pretrain_valid_dataset = train_valid_split(pretrain_train_dataset, 0.2)
    
    @property
    def checkpoints_dir(self) -> Path:
        return self.config.log_dir / ("checkpoints"+self.md5)
    
    @property
    def md5(self):
        from tasks.byol_task import BYOL_Task
        return hashlib.md5(str(self).encode('utf-8')+str(Classifier.HParams).encode('utf-8')+str(BYOL_Task.Options).encode('utf-8')).hexdigest()


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(TaskIncremental_Semi_Supervised, dest="experiment")

    args = parser.parse_args()
    experiment: TaskIncremental_Semi_Supervised = args.experiment
    experiment.launch()
