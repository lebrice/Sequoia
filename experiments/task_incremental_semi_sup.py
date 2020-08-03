import os
import hashlib
import tqdm

from timeit import default_timer as timer
from functools import partial
import itertools
import copy
from utils.logging_utils import get_logger
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate, cycle 
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Iterator, Callable
from torchvision.transforms import Compose
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

from datasets.data_utils import unbatch, unlabeled, get_semi_sampler, get_lab_unlab_idxs, train_valid_split, zip_dataloaders, SemiSupervisedDataset, BatchSampler_SemiUpservised
from datasets.subset import ClassSubset, Subset
from models.output_head import OutputHead
from simple_parsing import choice, field, list_field, mutable_field, subparsers
from simple_parsing.helpers import Serializable

from torch.utils.data.sampler import SubsetRandomSampler
from tasks import Tasks
from datasets.datasets import DatasetsHParams
from utils import utils
from experiments import TaskIncremental
from experiments.task_incremental import Modes
from utils.utils import n_consecutive, roundrobin

from .experiment import Experiment
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

logger = get_logger(__file__)
@dataclass 
class TaskIncremental_Semi_Supervised(TaskIncremental):
    """Task incremental semi-supervised setting
    """
    @dataclass  
    class Config(TaskIncremental.Config):
        #wether to apply simclr augment
        ratio_labelled: float = 1.
        #label incremental continual semi-supervised setting (scenario 2)
        label_incremental: bool = 0 

        #stationary unlabeled dataset - augment current task dataset with stationary unlabeled data (for scenario 3 - internet data)
        dataset_unlabeled: DatasetConfig = choice({
                d.name: d.value for d in Datasets
            }, default=Datasets.mnist.name)
        
        #whether to use 'dataset_unlabeled' (which is stationary)
        use_full_unlabeled: bool = False

        #reduce the size of the 'dataset_unlabeled' (stationary) to some amount of data per class
        reduce_full_unlabeled: float = 0.2

        datasets_dir: Path = Path(os.environ["HOME"]).joinpath('data')

        #baseline: at each time step train (semi-supervised) on data from all tasks sofar
        baseline_cl: bool = 0

    @dataclass
    class State(TaskIncremental.State):
        idx_lab_unlab: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)


    # Experiment Configuration. 
    config: Config = mutable_field(Config)     # Overwrite the type from Experiment.
    # Experiment state.
    state: State = mutable_field(State, init=False)        # Overwrite the type from Experiment.

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)

        self.fintuning_mlp = False #set to True when finetuning MLP on top of fixed encoder
        self.train_datasets_unlabeled: List[ClassSubset] = []
        self.epoch_length: Optional[int] = None
        self.batch_idx: Optional[int] = 0
        self.current_lr: Optional[float] = self.hparams.learning_rate
        self.full_train_dataset_unlabelled = None
        print(f'\n {torch.get_num_threads()} cpu cores available \n')
    
    def setup(self):
        super().setup()
        if self.state.global_step == 0:
            self.state.cumul_losses_linear_semi = [None for _ in range(self.n_tasks)] # [N]
        
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

    # def load_task_datasets(self, tasks: List[Task]) -> None:
    #     self.pretrain_train_dataset = None
    #     self.pretrain_valid_dataset = None

    #     self.train_datasets.clear()
    #     self.valid_datasets.clear()
    #     self.test_datasets.clear()

    #     datasets = self._load_task_datasets(tasks)

    #     (self.full_train_dataset, 
    #     self.full_valid_dataset, 
    #     self.full_test_dataset, 
    #     self.train_datasets, 
    #     self.train_datasets_unlabeled,
    #     self.valid_datasets, 
    #     self.test_datasets,  
    #     self.valid_cumul_datasets,
    #     self.test_cumul_datasets,
    #     self.full_train_dataset_unlabelled) = datasets

    def load_task_datasets(self, tasks: List[Task]): #, datasets: Tuple[Dataset, Dataset, Dataset ]=None, ratio_labelled:float=None):
            """Create the train, valid, test, as well as corresponding semi-samplers and cumulative valid & test datasets
            for each task.
            """
            #ratio_labelled = ratio_labelled if ratio_labelled is not None else self.config.ratio_labelled
            transform_train = None
            transform_test = None
            transform_valid = None    
            if self.config.simclr_augment_train and not self.config.simclr_augment_test :
                    from tasks.simclr.simclr_task_ptl import SimCLRTrainDataTransform_
                    transform_train = Compose([ToTensor(), ToPILImage(), SimCLRTrainDataTransform_(dobble=self.config.simclr_augment_train_dobble, input_height=self.config.dataset.x_shape[-1])])
                    transform_test = ToTensor()
                    transform_valid = ToTensor()
                    #in order not to effect validation dataset subset, which refers tot he same underlying dataset

            elif self.config.simclr_augment_train and self.config.simclr_augment_test:
                    from tasks.simclr.simclr_task_ptl import SimCLRTrainDataTransform_, SimCLREvalDataTransform_
                    transform_train = SimCLRTrainDataTransform_(dobble=self.config.simclr_augment_train_dobble, input_height=self.config.dataset.x_shape[-1])
                    transform_valid = SimCLREvalDataTransform_(dobble=self.config.simclr_augment_train_dobble, input_height=self.config.dataset.x_shape[-1])
                    transform_test = SimCLREvalDataTransform_(dobble=self.config.simclr_augment_train_dobble, input_height=self.config.dataset.x_shape[-1])
                    #in order not to effect validation dataset subset, which refers tot he same underlying dataset
            
            # if datasets is None:
            train_dataset, valid_dataset, test_dataset = self.load_datasets(train_transform =transform_train,valid_transform =transform_valid,test_transform = transform_test)
            # else:
            #     train_dataset, valid_dataset, test_dataset = datasets

            full_train_dataset_unlabelled = None
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
            self.train_datasets_unlabeled.clear()

            for i, task in enumerate(tasks):
                train = ClassSubset(train_dataset, task)
                valid = ClassSubset(valid_dataset, task)
                test  = ClassSubset(test_dataset, task)

                #get labeled and unlabeled indicies 
                idx_lab_new, idx_unlab_new = get_lab_unlab_idxs(train.targets, p=self.config.ratio_labelled)
                indices_train_lab, indices_train_unlab = self.state.idx_lab_unlab.setdefault(str(task.classes), (idx_lab_new, idx_unlab_new))

                self.train_datasets.append(Subset(train, indices_train_lab))
                # indices_train_lab, indices_train_unlab = indices_train_lab.tolist(), indices_train_unlab.tolist() 
                # if len(indices_train_unlab)>0:  
                #     self.train_datasets.append(SemiSupervisedDataset(train, frozenset(indices_train_lab), frozenset(indices_train_unlab)))
                #     #partial, we decide in 'run', wether to sample only labeled or mixture or only unlabeled
                #     self.train_samplers.append(partial(BatchSampler_SemiUpservised(indices_train_lab,indices_train_unlab, self.hparams.batch_size)))
                # else:
                #     #in order to have same amount of update steps per epoch (same epoch length) and sam ebatch size we use semi-dataset and sampler even if all data is labeled
                #     self.train_datasets.append(SemiSupervisedDataset(train, frozenset(indices_train_lab), frozenset(indices_train_lab)))
                #     self.train_samplers.append(partial(BatchSampler_SemiUpservised(indices_train_lab,indices_train_lab, self.hparams.batch_size)))
                    
                #     #self.train_datasets.append(train)
                
                self.train_samplers.append(None)

                if self.config.baseline_cl and i>0:
                    self.train_datasets_unlabeled.append(torch.utils.data.ConcatDataset([Subset(train, indices_train_unlab),self.train_datasets_unlabeled[i-1]]))
                else:
                    self.train_datasets_unlabeled.append(Subset(train, indices_train_unlab))

                self.valid_datasets.append(valid)
                self.test_datasets.append(test)
                self.test_samplers.append(None)
                self.valid_samplers.append(None)

                #set n_datapoints for a task (used in CRD distialtion loss)
                if task.n_data_points <0:
                    task.n_data_points = len(indices_train_unlab) + len(indices_train_lab)

            # Use itertools.accumulate to do the summation of the datasets.
            # self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
            # self.test_cumul_dataset = list(accumulate(self.test_datasets))
            #return (full_train_dataset, full_valid_dataset, full_test_dataset, train_datasets, train_datasets_unlabeled, valid_datasets, test_datasets, valid_cumul_datasets, test_cumul_dataset, full_train_dataset_unlabelled)

    def log_cumulative_loss(self, i):
        cumul_loss = self.state.cumul_losses[i]
        cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
        logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
        self.log({f"Cumulative": cumul_loss})

        cumul_loss_lin_full = self.state.cumul_losses_linear_full[i]             
        cumul_valid_accuracy_lin = get_supervised_accuracy(cumul_loss_lin_full, mode='Linear')
        logger.info(f"Cumul Accuracy linear [{i}]: {cumul_valid_accuracy_lin}")
        self.log({f"Cumulative_linear_full": cumul_loss_lin_full})

        cumul_loss_lin_semi = self.state.cumul_losses_linear_semi[i]            
        cumul_valid_accuracy_lin = get_supervised_accuracy(cumul_loss_lin_semi, mode='Linear')
        logger.info(f"Cumul Accuracy linear [{i}]: {cumul_valid_accuracy_lin}")
        self.log({f"Cumulative_linear_semi": cumul_loss_lin_semi})
    
    def measure_mlp_performance(self, i, j, train_j, test_j):    
            # Measure the "quality" of the representations, by training and
            # evaluating a classifier on train and test data from task J.       
            #We will use fully labeled data to train this classifier to stay comperable to fully supervised setting.
            if j==0:
                self.state.cumul_losses_linear_full[i]= LossInfo("Cumulative_lin_full")
                self.state.cumul_losses_linear_semi[i]= LossInfo("Cumulative_lin_semi")

            if isinstance(train_j, Subset):
                train_j_full = train_j.dataset
            else:
                train_j_full = train_j
            test_j_full = test_j

            linear_j_train_loss, linear_j_test_loss = self.evaluate_MLP(
                train_j_full,
                test_j_full,
                self.get_hidden_codes_array,
                description=f"Linear full [{i}][{j}]"
            )

            #Full
            linear_j_train_acc = get_supervised_accuracy(linear_j_train_loss)
            linear_j_test_acc = get_supervised_accuracy(linear_j_test_loss)
            logger.info(f"Task{i}: Linear Train Accuracy (Full) [{j}]: {linear_j_train_acc:.2%}")
            logger.info(f"Task{i}: Linear Test  Accuracy (Full) [{j}]: {linear_j_test_acc :.2%}")
            # Log the accuracies to wandb.
            self.log({
                f"Linear/train_full/task{j}": linear_j_train_acc,
                f"Linear/test_full/task{j}" : linear_j_test_acc,
            })   

            self.state.cumul_losses_linear_full[i].absorb(linear_j_test_loss)

            #Semi
            linear_j_train_loss, linear_j_test_loss = self.evaluate_MLP(
                train_j,
                test_j,
                self.get_hidden_codes_array,
                description=f"Linear semi [{i}][{j}]"
            )
            linear_j_train_acc = get_supervised_accuracy(linear_j_train_loss)
            linear_j_test_acc = get_supervised_accuracy(linear_j_test_loss)
            logger.info(f"Task{i}: Linear Train Accuracy [{j}]: {linear_j_train_acc:.2%}")
            logger.info(f"Task{i}: Linear Test  Accuracy [{j}]: {linear_j_test_acc :.2%}")
            # Log the accuracies to wandb.
            self.log({
                f"Linear/train/task{j}": linear_j_train_acc,
                f"Linear/test/task{j}" : linear_j_test_acc,
            })      
            self.state.cumul_losses_linear_semi[i].absorb(linear_j_test_loss)
    
    def train(self,
              train_dataloader: Union[Dataset, DataLoader, Iterator],                
              valid_dataloader: Union[Dataset, DataLoader, Iterator],    
              test_dataloader:Union[Dataset, DataLoader, Iterator],
              epochs: int,                
              description: str=None,
              mode: str = None,
              early_stopping_options: EarlyStoppingOptions=None,
              use_accuracy_as_metric: bool=None,                
              temp_save_dir: Path=None,
              steps_per_epoch: int = None,
              eval_function: Callable = None) -> TrainValidLosses:
        
        self.fintuning_mlp = False
        #steps_per_epoch = len(train_dataloader)  
        if mode == Modes.FINETUNE_CLS.value and not isinstance(train_dataloader, unlabeled):
            self.fintuning_mlp = True
        
        elif mode == Modes.PRETRAIN.value not in mode and not isinstance(train_dataloader, unlabeled):
            #mixture of labeled and unlabeled training
            
            if isinstance(train_dataloader, Dataset):
               train_dataloader = self.get_dataloader(train_dataloader)

            unlabeled_dataset = self.get_unlabeled_dataset()
            if len(unlabeled_dataset)>0:
                #labeled and unlabeled data is use for semoi-sup training 
                unlabeled_dataset = torch.utils.data.ConcatDataset([unlabeled_dataset, train_dataloader.dataset])
                train_dataloader_unlabeled = self.get_dataloader(unlabeled_dataset) #, batch_size=batch_size)
                train_dataloader = zip_dataloaders(train_dataloader_unlabeled,train_dataloader)
            else:
                #no unlabeled data, all data is labeled
                self.epoch_length = len(train_dataloader) 
                train_dataloader = zip_dataloaders(train_dataloader,train_dataloader)

        
        elif isinstance(train_dataloader, unlabeled):
            #only semi-supervised training (no supervision)

            #concat given unlabeled dataset (train) with unlabeled data (train_dataset_unlabeled)
            unlabeled_dataset = self.get_unlabeled_dataset()
            full_dataset_task = torch.utils.data.ConcatDataset([train_dataloader.loader.dataset, unlabeled_dataset])
            train_dataloader = unlabeled(self.get_dataloader(full_dataset_task))
            #self.epoch_length = len(train_dataloader)
            #steps_per_epoch = self.epoch_length

        self.epoch_length = len(train_dataloader)
        return super().train(train_dataloader, valid_dataloader, test_dataloader, 
                            epochs, description, mode, early_stopping_options, use_accuracy_as_metric,
                            temp_save_dir, steps_per_epoch, eval_function)
    
    def get_unlabeled_dataset(self) -> Dataset:
        if self.config.label_incremental:
            #LI scenario
            #always use full dataset for self-sup learning (e.g. full cifar 100)
            if self.full_train_dataset_unlabelled is None:
                #we are in LI
                return self.full_train_dataset
            else:
                #we are in LI + internet data scenario (we also use some additional data from internet)
                logger.info("Using full unlabeled set")
                full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.full_train_dataset,self.full_train_dataset_unlabelled])
                return full_unlabeled_dataset
        else:
            if self.full_train_dataset_unlabelled is None:
                #Fully incremental scenario
                return self.train_datasets_unlabeled[self.state.i]
            else:
                #FI + internet data
                logger.info("Using full unlabeled set")
                full_unlabeled_dataset = torch.utils.data.ConcatDataset([self.train_datasets_unlabeled[self.state.i],self.full_train_dataset_unlabelled])
                return full_unlabeled_dataset
    
    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.batch_idx +=1 
        self.model.train()   
        for batch in dataloader: 
            if len(batch)!=2: #unsupervised batch
                data, target = self.preprocess(batch)
            elif len(batch[0])==2 and len(batch[0])==len(batch[1]): # semi-supervised batch
                batch_sup, batch_unsup = batch
                data, target = self.preprocess(batch_sup)
                u, _ = self.preprocess(batch_unsup)
                #create mixed batch 
                 
                data = torch.cat([data, u], dim=0)
                if target is not None:
                    target = torch.stack(list(target)+([torch.tensor(-1).to(self.model.out_device)]*len(u)))
            else: #fully supervised
                data, target = self.preprocess(batch)

            yield self.train_batch(data, target)

    def train_batch(self, *args, **kwargs) -> LossInfo:
        if not self.fintuning_mlp:
            loss = super().train_batch(*args, **kwargs)
        else:
            loss = self.train_batch_MLP(*args, **kwargs)
        loss = self.state.perf_meter.update(loss)
        return loss

        
    def train_batch_MLP(self, data: Tensor, target: Tensor, name: str="Train_MLP") -> LossInfo:
        self.model.train()
        self.model.optimizer.zero_grad()
        
        loss = LossInfo(name)   
        #start = timer()     
        batch_loss_info_supervised = self.model.supervised_loss(data, target)
        #end = timer()
        #print("supervised loss ",end - start) 

        loss.total_loss = torch.zeros(1, device=batch_loss_info_supervised.total_loss.device)    
        loss += batch_loss_info_supervised

        total_loss = batch_loss_info_supervised.total_loss
        total_loss.backward()

        self.step(global_step=self.global_step)
        self.global_step += data.shape[0]

        return loss

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
    
    
if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()    
    parser.add_arguments(TaskIncremental_Semi_Supervised, dest="experiment")

    #from datasets.datasets import DatasetsHParams
    #parser.add_arguments(DatasetsHParams, "options")

    args = parser.parse_args()    
    experiment: TaskIncremental_Semi_Supervised = args.experiment
    experiment.launch()
