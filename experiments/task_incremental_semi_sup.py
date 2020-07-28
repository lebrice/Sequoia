import os
import hashlib
import tqdm
import itertools
import copy
from utils.logging_utils import get_logger
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate, cycle 
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Iterator
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

from datasets.data_utils import unbatch, unlabeled, get_semi_sampler, get_lab_unlab_idxs, train_valid_split
from datasets.subset import ClassSubset, Subset
from models.output_head import OutputHead
from simple_parsing import choice, field, list_field, mutable_field, subparsers
from simple_parsing.helpers import Serializable

from torch.utils.data.sampler import SubsetRandomSampler
from tasks import Tasks
from datasets.datasets import DatasetsHParams
from utils import utils
from experiments import TaskIncremental
from utils.utils import n_consecutive, roundrobin

from .experiment import Experiment
from tasks.simclr.simclr_task import SimCLRTask
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

class RescaleTarget(object):
    def __init__(self, min_class_label):
        self.min_class_label = min_class_label
    def __call__(self, y):
        return y-self.min_class_label
    def __repr__(self):
        return self.__class__.__name__ + '()'  

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
        #stationary unlabeled dataset
        dataset_unlabeled: DatasetConfig = choice({
                d.name: d.value for d in Datasets
            }, default=Datasets.mnist.name)
        #use full unlabaled dataset 
        use_full_unlabeled: bool = False

        reduce_full_unlabeled: float = 0.2
        datasets_dir: Path = Path(os.environ["HOME"]).joinpath('data')

        #baseline: at each time step train (semi-) on data from all tasks sofar
        baseline_cl: bool = 0

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

        print(f'\n {torch.get_num_threads()} cpu cores available \n')

    def run(self):
        """Evaluates a model/method in the classical "task-incremental" setting.

        NOTE: We evaluate the performance on all tasks:
        - When the task has NOT been trained on before, we evaluate the ability
        of the representations to "generalize" to unseen tasks by training a KNN
        classifier on the representations of target task's training set, and
        evaluating it on the representations of the target task's validation set.
        - When the task has been previously trained on, we evaluate the
        classification loss/metrics (and auxiliary tasks, if any) as well as the
        representations with the KNN classifier.

        Roughly equivalent to the following pseudocode:
        ```
        # Training and Validdation datasets
        train_datasets: Dataset[n_tasks]
        valid_datasets: Dataset[n_tasks]

        # Arrays containing the loss/performance metrics. (Value at idx (i, j)
        # is the preformance on task j after having trained on tasks [0:i].)
        
        knn_losses: LossInfo[n_tasks][n_tasks] 
        tasks_losses: LossInfo[n_tasks][j]  #(this is a lower triangular matrix)

        # Array of objects containing the loss/performance on tasks seen so far.
        cumul_losses: LossInfo[n_tasks]
        
        for i in range(n_tasks):
            train_until_convergence(train_datasets[i], valid_datasets[i])
            
            # Cumulative (supervised) validation performance.
            cumul_loss = LossInfo()

            for j in range(n_tasks):
                # Evaluate the representations with a KNN classifier.
                knn_loss_j = evaluate_knn(train_dataset[j], valid_datasets[j])
                knn_losses[i][j] = knn_loss_j
                
                if j <= i:
                    # We have previously trained on this class.
                    loss_j = evaluate(valid_datasets[j])
                    task_losses[i][j] = loss_j
                    cumul_loss += loss_j

            cumul_losses[i] = cumul_loss
        ```
        """
        self.setup()          
        self.load_pretrain_dataset()
        self.pretrain_unsup()
        self.load_task_datasets(self.tasks)
                
        for i in range(self.state.i, self.n_tasks):
            self.state.i = i
            logger.info(f"Starting task {i} with classes {self.tasks[i]}")
            self.log({"task/currently_learned_task": self.state.i})

            self.on_task_switch(self.tasks[i])
            # Training and validation datasets for task i.
            train_i_dataset = self.train_datasets[i]
            valid_i_dataset = self.valid_datasets[i]
            test_i_dataset = self.test_datasets[i]            
            train_i_dataset_unl = self.train_datasets_unlabeled[i]

            if self.replay_buffer:
                # Append the replay buffer to the end of the training dataset.
                # TODO: Should we shuffle them together?
                # TODO: Should we also add some data from previous tasks in the validation dataset?
                train_i_dataset += self.replay_buffer.as_dataset()

            train_i_loader = self.get_dataloader(train_i_dataset)
            valid_i_loader = self.get_dataloader(valid_i_dataset)
            test_i_loader = self.get_dataloader(test_i_dataset)

            if self.state.j == 0:
                with self.plot_region_name(f"Learn Task {i}"):
                    # We only train (unsupervised) if there is at least one enabled
                    # auxiliary task and if the maximum number of unsupervised
                    # epochs per task is greater than zero.
                    self_supervision_on = any(task.enabled for task in self.model.tasks.values())

                    if self_supervision_on and self.config.unsupervised_epochs_per_task:
                        # Un/self-supervised training on task i.
                        # NOTE: Here we use the same dataloaders, but drop the
                        # labels and treat them as unlabeled datasets. 
                        self.state.all_losses += self.train(
                            unlabeled(self.get_dataloader(torch.utils.data.ConcatDataset([train_i_dataset_unl, train_i_dataset]))),
                            unlabeled(valid_i_loader),
                            None,
                            epochs=self.config.unsupervised_epochs_per_task,
                            description=f"Task {i} (Unsupervised)",
                            use_accuracy_as_metric=False, # Can't use accuracy as metric during unsupervised training.
                            temp_save_dir=self.checkpoints_dir / f"task_{i}_unsupervised",
                        )

                    # Train (supervised) on task i.
                    # TODO: save the state during training?.                    
                    self.state.all_losses += self.train(
                        train_i_loader,
                        valid_i_loader,
                        test_i_loader,
                        epochs=self.config.supervised_epochs_per_task,
                        description=f"Task {i} (Supervised)",
                        use_accuracy_as_metric=self.config.use_accuracy_as_metric,
                        temp_save_dir=self.checkpoints_dir / f"task_{i}_supervised",
                    )
            
            # Save to the 'checkpoints' dir
            self.save_state()
            
            # Evaluation loop:
            for j in range(self.state.j, self.n_tasks):
                if j == 0:
                    #  Evaluate on all tasks (as described above).
                    if self.state.cumul_losses[i] is not None:
                        logger.warning(
                            f"Cumul loss at index {i} should have been None "
                            f"but is {self.state.cumul_losses[i]}.\n"
                            f"This value will be overwritten."
                        )
                    self.state.cumul_losses[i] = LossInfo("Cumulative")

                self.state.j = j

                if j <= i:
                    # -- Evaluate Representations after having learned tasks [0:i] on data from task J. --
                    try:

                        train_j = self.train_datasets[j]
                        test_j  = self.test_datasets[j]   

                        # If we have previously trained on this task:
                        #self.on_task_switch(self.tasks[j])
                        self.model.current_task = self.tasks[j]

                        # Test on the test dataset for task j.
                        loss_j = self.test(test_j, description=f"Task{i}: Test on Task{j}", name=f"Task{j}")
                        self.log({f"Task_losses/Task{j}": loss_j})
                        
                        if j==i:
                            self.log({f"Test_full_Final": loss_j})
                        
                        self.state.task_losses[i][j] = loss_j
                        # Merge the metrics from this task and the other tasks.
                        # NOTE: using += above would add a "Task<j>" item in the
                        # `losses` attribute of the cumulative loss, without merging
                        # the metrics.
                        self.state.cumul_losses[i].absorb(loss_j)

                        supervised_acc_j = get_supervised_accuracy(loss_j)
                        logger.info(f"Task {i} Supervised Test accuracy on task {j}: {supervised_acc_j:.2%}")

                        # Measure the "quality" of the representations, by training and
                        # evaluating a classifier on train and test data from task J.
                        linear_j_train_loss, linear_j_test_loss = self.evaluate_MLP(
                            train_j,
                            test_j,
                            description=f"Linear [{i}][{j}]"
                        )
                        linear_j_train_acc = linear_j_train_loss.metric.accuracy
                        linear_j_test_acc = linear_j_test_loss.metric.accuracy
                        logger.info(f"Task{i}: Linear Train Accuracy [{j}]: {linear_j_train_acc:.2%}")
                        logger.info(f"Task{i}: Linear Test  Accuracy [{j}]: {linear_j_test_acc :.2%}")
                        # Log the accuracies to wandb.
                        self.log({
                            f"Linear/train/task{j}": linear_j_train_acc,
                            f"Linear/test/task{j}" : linear_j_test_acc,
                        })

                    
                        
                    except Exception as e:
                        print(e)
                
            # -- Evaluate representations after task i on the whole train/test datasets. --

            # Measure the "quality" of the representations of the data using the
            # whole test dataset.
            # TODO: Do this in another process (as it might take very long)
            try:
                knn_train_loss, knn_test_loss = self.test_knn(
                    self.full_train_dataset,
                    self.full_test_dataset,
                    description=f"Task{i}: KNN (Full Dataset)"
                )

                knn_train_acc = knn_train_loss.accuracy
                knn_test_acc  = knn_test_loss.accuracy
                logger.info(f"Task{i}: KNN Train Accuracy (Full): {knn_train_acc:.2%}")
                logger.info(f"Task{i}: KNN Test  Accuracy (Full): {knn_test_acc :.2%}")
                # Log the accuracies to wandb.
                self.log({
                    f"KNN/train/full": knn_train_acc,
                    f"KNN/test/full" : knn_test_acc,
                })
                # Save the loss object in the state.
                self.state.knn_full_losses[i] = knn_test_loss
            except Exception as e:
                        print(e)

            # Save the state with the new metrics, but no need to save the
            # model weights, as they didn't change.
            self.save_state(save_model_weights=False)
            # NOTE: this has to be after, so we don't reloop through the j's if
            # something in the next few lines fails.
            self.state.j = 0            
            cumul_loss = self.state.cumul_losses[i]            
            cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
            logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
            self.log({f"Cumulative": cumul_loss})

        # mark that we're done so we get right back here if we resume a
        # finished experiment
        self.state.i = self.n_tasks
        self.state.j = self.n_tasks

        self.save_state(self.checkpoints_dir) # Save to the 'checkpoints' dir.
        self.save_state(self.results_dir) # Save to the 'results' dir.
        
        for i, cumul_loss in enumerate(self.state.cumul_losses):
            assert cumul_loss is not None, f"cumul loss at {i} should not be None!"
            cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
            logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
            if self.config.use_wandb:
                wandb.run.summary[f"Cumul Accuracy [{i}]"] = cumul_valid_accuracy

        from utils.plotting import maximize_figure
        # Make the forward-backward transfer grid figure.
        #grid = self.make_transfer_grid_figure(
        #    knn_losses=self.state.knn_losses,
        #    task_losses=self.state.task_losses,
        #    cumul_losses=self.state.cumul_losses
        #)
        #grid.savefig(self.plots_dir / "transfer_grid.png")
        
        # if self.config.debug:
        #     grid.waitforbuttonpress(10)

        # make the plot of the losses (might not be useful, since we could also just do it in wandb).
        # fig = self.make_loss_figure(self.all_losses, self.plot_sections)
        # fig.savefig(self.plots_dir / "losses.png")
        
        # if self.config.debug:
        #     fig.show()
        #     fig.waitforbuttonpress(10)

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
            
            train_dataset, valid_dataset, test_dataset = self.load_datasets(train_transform =transform_train,valid_transform =transform_valid,test_transform = transform_test)

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
                # if self.model.hparams.multihead:
                #     target_transform = RescaleTarget(min(task.classes))
                # else:
                #     target_transform = None
                #train_dataset_ = copy.deepcopy(train_dataset)
                train = ClassSubset(train_dataset, task)
                #train.dataset.dataset.target_transform = target_transform

                #valid_dataset_ = copy.deepcopy(valid_dataset)
                valid = ClassSubset(valid_dataset, task)
                #valid.dataset.dataset.target_transform = target_transform

                #test_dataset_ = copy.deepcopy(test_dataset)
                test  = ClassSubset(test_dataset, task)
                #test.dataset.target_transform = target_transform

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
                if self.config.baseline_cl and i>0:
                    self.train_datasets_unlabeled.append(torch.utils.data.ConcatDataset([Subset(train, indices_train_unlab),self.train_datasets_unlabeled[i-1]]))
                
                else:
                    self.train_datasets_unlabeled.append(Subset(train, indices_train_unlab))
                self.valid_datasets.append(valid)
                self.test_datasets.append(test)

                if task.n_data_points <0:
                    task.n_data_points = len(indices_train_unlab) + len(indices_train_lab)

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

        self.epoch_length = len(train_dataloader)
        steps_per_epoch = len(train_dataloader)
        if not 'pretrain' in description and not isinstance(train_dataloader, unlabeled) and self.config.ratio_labelled<1: 
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
                 
                data = torch.cat([data, u], dim=0)
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

    # def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor], Tuple[Tuple[Tensor,Tensor], Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
    #     if len(batch)==1:
    #         batch = (torch.stack(batch[0]).transpose(1,0),)
    #     elif isinstance(batch[0], tuple) or isinstance(batch[0], list):
    #         batch[0] = torch.stack(batch[0]).transpose(1,0)
    #     data, target = super().preprocess(batch)
    #     #if self.config.simclr_augment: 
    #     #    data, target = SimCLRTask.preprocess_simclr(data, target)
    #     return data, target


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
