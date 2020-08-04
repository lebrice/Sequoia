import itertools
import hashlib
import copy
import traceback
from enum import Enum

import torch.multiprocessing as mp
from multiprocessing import Value
from utils.logging_utils import get_logger  
from collections import OrderedDict, defaultdict, Sized
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Iterator
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import Tensor
from datasets.data_utils import train_valid_split, TransformedDataset
from models.classifier import Classifier
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image
from datasets import Datasets
from common.losses import (LossInfo, TrainValidLosses, get_supervised_accuracy,
                           get_supervised_metrics, AUC_Meter)
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics, AROC
from common.task import Task
from datasets import DatasetConfig
from datasets.data_utils import unbatch, unlabeled
from datasets.subset import ClassSubset
from models.output_head import OutputHead
from simple_parsing import choice, field, list_field, mutable_field, subparsers
from utils.json_utils import Serializable
from tasks import Tasks
from utils import utils
from utils.utils import n_consecutive, roundrobin
from tasks.simclr.simclr_task_ptl import SimCLRTrainDataTransform_, SimCLREvalDataTransform_
from torchvision.transforms import Compose, ToPILImage, ToTensor
from utils.eval_utils import background_eval

from .experiment import Experiment

logger = get_logger(__file__)

class Modes(Enum):
    UNSUP: str = 'unsupervised'
    SUP: str = 'supervised'
    PRETRAIN: str = 'pretrain'
    FINETUNE_CLS: str = 'fintuning_classifier'



@dataclass
class TaskIncremental(Experiment):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    @dataclass
    class Config(Experiment.Config):
        """ Configuration options for the TaskIncremental experiment. """
        # Number of classes per task.
        n_classes_per_task: int = 2

        #number of tasks
        n_tasks: int = 5             
        # Wether to sort out the classes in the class_incremental setting.
        random_class_ordering: bool = False
        # Maximum number of epochs of self-supervised training to perform on the
        # task data before switching to supervised training.
        unsupervised_epochs_per_task: int = 1
        # Maximum number of epochs of supervised training to perform on each task's
        # dataset.
        supervised_epochs_per_task: int = 0

        # Maximum number of epochs of supervised finetuning (classifier only!!!) to perform on each task's
        # dataset.
        finetune_epochs_per_task: int = 1

        #If `True`, accuracy will be used as a measure of performance. Otherwise, the total validation loss is used. Defaults to False.
        use_accuracy_as_metric: bool = False

        task_labels_at_train_time: bool = True
        task_labels_at_test_time:  bool = True

        #number of self-sup pretraining epochs
        unsupervised_epochs_pretraining: int = 0

        #pretraining dataset
        pretraining_dataset: DatasetConfig = choice({
                d.name: d.value for d in Datasets
            }, default=Datasets.mnist.name)

        #reinitialize after each task
        reinit_each_step: bool = 0
        #wether to apply simclr augmentation to training set
        simclr_augment_train: bool = 1
        #wether to apply simclr dobble  augmentation to training set (for supervised baseline with augmentation better to set this to 0)
        simclr_augment_dobble: bool = 1

        #wether to apply simclr augmentation to test set
        simclr_augment_test: bool = 1

        #MLP finetuning LR
        finetuning_lr: float =0.0001 

        #path to a folder to try to load best model from (if None no mdoel is loaded)
        model_path: Optional[str] = None #"/Users/oleksostapenko/Projects/SSCL/results/SSCL/TaskIncremental_Semi_Supervised/EkYwW/checkpoints"


    @dataclass
    class State(Experiment.State):
        """Object that contains all the state we want to be able to save/restore.

        We aren't going to parse these from the command-line.
        """
        tasks: List[Task] = list_field()

        i: int = 0
        j: int = 0

        # Container for the losses. At index [i, j], gives the validation
        # metrics on task j after having trained on tasks [0:i], for 0 < j <= i.
        task_losses: List[List[Optional[LossInfo]]] = list_field()

        # Container for the KNN metrics. At index [i, j], gives the accuracy of
        # a KNN classifier trained on the representations of the samples from
        # task [i], evaluated on the representations of the the samples of task j.
        # NOTE: The representations for task i are obtained using the encoder
        # which was trained on tasks 0 through i.
        knn_losses: List[List[LossInfo]] = list_field()

        # Knn Losses on the entire test set after having learned each task.
        knn_full_losses: List[LossInfo] = list_field()

        # Cumulative losses after each task
        cumul_losses: List[Optional[LossInfo]] = list_field()


        perf_meter: AUC_Meter = mutable_field(AUC_Meter, repr=False, init=False)

        epoch:int = 0  

        # Experiment Configuration.
    config: Config = mutable_field(Config)     # Overwrite the type from Experiment.
    # Experiment state.
    state: State = mutable_field(State, init=False)        # Overwrite the type from Experiment.

    def __post_init__(self, *args, **kwargs):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__(*args, **kwargs)
        self.config: TaskIncremental.Config
        # The entire training, validation and testing datasets.
        self.full_train_dataset : VisionDataset = None
        self.full_valid_dataset : VisionDataset = None
        self.full_test_dataset  : VisionDataset = None
        self.batch_idx: Optional[int] = 0
        self.epoch_length: Optional[int] = None
        #pretraining datasets
        self.pretrain_train_dataset: VisionDataset = None
        self.pretrain_valid_dataset: VisionDataset = None

        # Datasets for each task
        self.train_datasets: List[ClassSubset] = []
        self.valid_datasets: List[ClassSubset] = []
        self.test_datasets: List[ClassSubset] = []

        #samplers
        self.train_samplers: List[SubsetRandomSampler] = []
        self.valid_samplers: List[SubsetRandomSampler] = []
        self.test_samplers: List[SubsetRandomSampler] = []



        # Cumulative datasets: Hold the data from previously seen tasks
        self.valid_cumul_datasets: List[ClassSubset] = []
        self.test_cumul_datasets: List[ClassSubset] = []

        # Keeps track of which tasks have some data stored in the replay buffer
        # (if using replay)
        self.tasks_in_buffer: List[Task] = []

        #Queues for linear evaluations
        self.background_mlp_queue = mp.Queue()
        self.background_process = mp.Process(target=background_eval, args=(self.background_mlp_queue, self.config))
        self.background_process.start()
    
    def setup(self):
        """Prepare everything before training begins: Saves/restores state,
        loads the datasets, model weights, etc.
        """
        super().setup()
        if self.state.global_step == 0:
            logger.info("Starting from scratch!")   
            self.state.tasks = self.create_tasks_for_dataset(self.config.dataset)
        else:
            logger.info(f"Starting from global step {self.state.global_step}")
            logger.info(f"i={self.state.i}, j={self.state.j}")
        
        self.tasks = self.state.tasks
        logger.info(f"Class Ordering: {[t.classes for t in self.state.tasks]}")
        
        # save the state, just in case.
        self.save_state(save_model_weights=False)

        # Load the datasets
        if self.config.n_tasks==0:
            self.n_tasks = len(self.tasks) 
        else:
            self.n_tasks = self.config.n_tasks

        
        if self.state.global_step == 0:
            self.state.knn_losses   = [[None for _ in range(self.n_tasks)] for _ in range(self.n_tasks)]
            self.state.knn_full_losses = [None for _ in range(self.n_tasks)]
            self.state.task_losses  = [[None for _ in range(i+1)] for i in range(self.n_tasks)] # [N,J]
            self.state.cumul_losses = [None for _ in range(self.n_tasks)] # [N]
            self.state.cumul_losses_linear_full = [None for _ in range(self.n_tasks)] # [N]

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
        try:        
            for i in range(self.state.i, self.n_tasks):
                self.state.i = i
                logger.info(f"Starting task {i} with classes {self.tasks[i]}")
                if self.config.model_path is not None:
                    self.try_loading_best_model_unsupervised(task=i)

                self.log({"task/currently_learned_task": self.state.i})

                self.on_task_switch(self.tasks[i])
                # Training and validation datasets for task i.
                train_i_dataset = self.train_datasets[i]
                valid_i_dataset = self.valid_datasets[i]
                test_i_dataset = self.test_datasets[i]

                train_i_sampler = self.train_samplers[i]
                valid_i_sampler = self.valid_samplers[i]
                test_i_sampler = self.test_samplers[i]            


                if self.replay_buffer:
                    # Append the replay buffer to the end of the training dataset.
                    # TODO: Should we shuffle them together?
                    # TODO: Should we also add some data from previous tasks in the validation dataset?
                    train_i_dataset += self.replay_buffer.as_dataset()

                train_i_loader = self.get_dataloader(train_i_dataset, sampler = train_i_sampler)
                valid_i_loader = self.get_dataloader(valid_i_dataset, sampler = valid_i_sampler)
                test_i_loader = self.get_dataloader(test_i_dataset, sampler = test_i_sampler)

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
                                unlabeled(train_i_loader),
                                unlabeled(valid_i_loader),
                            test_i_loader,
                                epochs=self.config.unsupervised_epochs_per_task,
                                description=f"Task {i}",
                                mode = Modes.UNSUP.value,
                                use_accuracy_as_metric=False, # Can't use accuracy as metric during unsupervised training.
                                temp_save_dir=self.checkpoints_dir / f"task_{i}_unsupervised",
                                eval_function=self.evaluate_mlp_baground,
                            )

                        # Train (supervised) on task i.
                        # TODO: save the state during training?.
                        self.state.all_losses += self.train(
                            train_i_loader,
                            valid_i_loader,
                            test_i_loader,
                            epochs=self.config.supervised_epochs_per_task,
                            description=f"Task {i}",
                            mode = Modes.SUP.value,
                            use_accuracy_as_metric=self.config.use_accuracy_as_metric,
                            temp_save_dir=self.checkpoints_dir / f"task_{i}_supervised",
                        )

                        # Finetune classifier (supervised) on task i.
                        # TODO: save the state during training?.
                        
                        self.set_optimizer_lr(self.config.finetuning_lr)
                        self.state.all_losses += self.train(
                            train_i_loader,
                            valid_i_loader,
                            test_i_loader,  
                            epochs=self.config.finetune_epochs_per_task,
                            description=f"Task {i}",
                            mode = Modes.FINETUNE_CLS.value,
                            use_accuracy_as_metric=self.config.use_accuracy_as_metric,
                            temp_save_dir=self.checkpoints_dir / f"task_{i}_supervised",
                        )
                        self.set_optimizer_lr(self.hparams.learning_rate)
                
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
                    # -- Evaluate Representations after having learned tasks [0:i] on data from task J. --
                    # If we have previously trained on this task:
                    if j<=i:
                        try:
                            train_j = self.train_datasets[j]
                            test_j  = self.test_datasets[j]     
                            train_j_sampler = self.train_samplers[j]   
                            test_j_sampler = self.test_samplers[j]   
                            train_loadder_j = self.get_dataloader(train_j, sampler=train_j_sampler)
                            test_loadder_j = self.get_dataloader(test_j, sampler=test_j_sampler)
                            
                            #self.on_task_switch(self.tasks[j])
                            self.model.current_task = self.tasks[j]

                            # Test on the test dataset for task j.
                            loss_j = self.test(test_loadder_j, description=f"Task{i}: Test on Task{j}", name=f"Task{j}")
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

                            self.measure_mlp_performance(i, j, train_j, test_j)


                        except Exception as e:
                            print(e)
                    
                # -- Evaluate representations after task i on the whole train/test datasets. --

                # Measure the "quality" of the representations of the data using the
                # whole test dataset.
                # TODO: Do this in another process (as it might take very long)
                knn_train_loss, knn_test_loss = self.test_knn(
                    self.full_train_dataset,
                    self.full_test_dataset,
                    description=f"Task{i}: KNN (Full Test Dataset)"
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

                # Save the state with the new metrics, but no need to save the
                # model weights, as they didn't change.
                self.save_state(save_model_weights=False)
                # NOTE: this has to be after, so we don't reloop through the j's if
                # something in the next few lines fails.
                self.state.j = 0       
                self.log_cumulative_loss(i)

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
            #     fig.waitforbuttonpress(10
        
            self.background_mlp_queue.put(None)
            self.background_process.join()
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("waiting for queue to finish...")
            self.background_queue.put(None)
            self.background_process.join()
            raise e
    
    def try_loading_best_model_unsupervised(self, task:int):
        path = Path(self.config.model_path).joinpath(self.get_model_name(mode=Modes.UNSUP.value, task=task))
        logger.info(f'Trying to load best model for task {task} from {path}')
        self.try_loading_model(path)
    
    def try_loading_model(self, path:Path):
        try:
            self.model.load_state_dict(torch.load(path))
        except Exception as e:
            print(e)
        
    def log_cumulative_loss(self, i):
        cumul_loss = self.state.cumul_losses[i]            
        cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
        logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
        self.log({f"Cumulative": cumul_loss})

        cumul_loss_lin = self.state.cumul_losses_linear_full[i]            
        cumul_valid_accuracy_lin = get_supervised_accuracy(cumul_loss_lin)
        logger.info(f"Cumul Accuracy linear [{i}]: {cumul_valid_accuracy_lin}")
        self.log({f"Cumulative_linear_full": cumul_loss_lin})
    
    def evaluate_mlp_baground(self, i, train_loader, test_loader, step, description) -> None:
        #get labeled dataloader 
               
        def get_dataset(dataloader):
            if isinstance(dataloader, unlabeled):
                dataloader = dataloader.loader
            if isinstance(dataloader, DataLoader):
                dataset = dataloader.dataset
            elif isinstance(dataloader, Dataset):
                dataset = dataloader
            else:
                raise Exception('Unrecognized object as dataloader')
            if isinstance(dataset, TransformedDataset):
                dataset = dataset.dataset
            return dataset

        #NOTE: potential sampler is ignored here (sampler is currentlhy not used)
        train_dataset = get_dataset(train_loader)
        test_dataset = get_dataset(test_loader)

        train_dataset = TransformedDataset(train_dataset, transform=SimCLRTrainDataTransform_(dobble=0,input_height=self.config.dataset.x_shape[-1]))
        test_dataset = TransformedDataset(test_dataset, transform=ToTensor())

        X, y = self.get_hidden_codes_array(train_loader)
        Xt, yt = self.get_hidden_codes_array(test_loader)

        self.background_mlp_queue.put((i, X, y, Xt, yt, step, description))
        
    @torch.no_grad()
    def get_hidden_codes_array(self, dataloader: DataLoader, rescale_y = True) -> Tuple[np.ndarray, np.ndarray]:
            """ Gets the hidden vectors and corresponding labels. """
            was_training = self.model.training
            self.model.eval()
            h_x_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            for batch in dataloader:
                x, y = self.preprocess(batch)
                if rescale_y:
                    y = self.model.rescale_target(y)
                # We only do KNN with examples that have a label.
                if y is not None:                    
                    x, y, _ = self.model.preprocess_inputs(x,y, None)
                    h_x = self.model.encode(x)
                    h_x_list.append(h_x.detach().cpu().numpy())
                    y_list.append(y.detach().cpu().numpy())
            if was_training:
                self.model.train()
            return np.concatenate(h_x_list), np.concatenate(y_list)
    
    def measure_mlp_performance(self, i, j, train_loader_j, test_loader_j):
        if j==0:
            self.state.cumul_losses_linear_full[i]= LossInfo("Cumulative_lin_full")

        # Measure the "quality" of the representations, by training and
        # evaluating a classifier on train and test data from task J.        
        linear_j_train_loss, linear_j_test_loss = self.evaluate_MLP(
            train_loader_j,
            test_loader_j,
            self.get_hidden_codes_array,
            description=f"Linear [{i}][{j}]"
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
        #to calculate average acc. over tasks
        self.state.cumul_losses_linear_full[i].absorb(linear_j_test_loss)

    def pretrain_unsup(self):
        if self.config.unsupervised_epochs_pretraining > 0:
            logger.info(f'\n ----Start pretraining on {self.config.pretraining_dataset.name} -----')
            pretrain_train_loader = self.get_dataloader(self.pretrain_train_dataset)
            pretrain_valid_loader = self.get_dataloader(self.pretrain_valid_dataset)
            self.train(
                unlabeled(pretrain_train_loader), 
                unlabeled(pretrain_valid_loader), 
                None, 
                self.config.unsupervised_epochs_pretraining,
                'pretrain',
                use_accuracy_as_metric=False)
            save_path = self.get_pretrained_model_path(self.model)
            self.save_weights(save_path)
            logger.info(f'Pretraining finished, model stored at {save_path}')

            if self.model.hparams.freeze_pretrained_model:
                    self.model = self.freeze_encoder(self.model)
    
    def train(self,train_dataloader: Union[Dataset, DataLoader, Iterator], *args, **kwargs) -> TrainValidLosses:
              if isinstance(train_dataloader, Sized): 
                self.epoch_length = len(train_dataloader)
              return super().train(train_dataloader, *args, **kwargs)

    def train_epoch(self, epoch, *args, **kwargs):
        self.state.epoch = epoch
        self.batch_idx = 0 
        return super().train_epoch(epoch, *args, **kwargs)

    def on_task_switch(self, task: Optional[Task], **kwargs) -> None:
        """Called when a task boundary is reached.

        TODO: @lebrice Adding this method on the `Experiment` class in
        case some addons need it?
        TODO: Maybe we could use `.on_task_switch(<None>)` when you want to
        inform the model that there is a task boundary, without giving the task
        labels.
        
        Args:
            task (Optional[Task]): If given, holds the information about the
            new task: an index which indicates in which order it was encountered,
            as well as the set of classes present in the new data. 
        """
        if self.model.training and not self.config.task_labels_at_train_time:
            logger.warning(f"Ignoring task label {task} since model is training and we don't have access to task labels at train time.")
            return
        elif not self.model.training and not self.config.task_labels_at_test_time:
            logger.warning(f"Ignoring task label {task} since model is in evaluation mode and we don't have access to task labels at test time.")
            return
        
        # If we are using a multihead model, we give it the task label (so
        # that it can spawn / reuse the output head for the given task).

        i = self.state.i
        ewc_task = self.model.tasks.get(Tasks.EWC)

        train_loader = self.get_dataloader(self.train_datasets[i], sampler = self.train_samplers[i])

        if task and self.replay_buffer is not None:
            self.update_replay_buffer(task, new_task_loader=train_loader)

        if ewc_task and ewc_task.enabled:
            prev_task = None if i == 0 else self.tasks[i-1]
            classifier_head: Optional[OutputHead] = None
            if i == 0:
                # TODO: @lebrice I don't quite recall why this is here..
                classifier_head = None
            elif self.model.hparams.multihead:
                classifier_head = self.model.get_output_head(prev_task)
            else:
                classifier_head = self.model.default_output_head

            if i != 0 and ewc_task.current_task_loader is None:
                previous_task_loader = self.get_dataloader(self.train_datasets[i-1])
                ewc_task.current_task_loader = previous_task_loader

            kwargs.setdefault("prev_task", prev_task)
            kwargs.setdefault("classifier_head", classifier_head)
            kwargs.setdefault("train_loader", train_loader)

        self.model.on_task_switch(task, **kwargs)
        if self.config.reinit_each_step and self.state.i>0:    
            self.model.reset_encoder_parameters()

    def update_replay_buffer(self, new_task: Task, new_task_loader: DataLoader) -> None:
        """Update the replay buffer, adding data for the new task.

        Which samples are added to the buffer can be controlled with another
        method. By default, we just choose an even number of samples for each
        class.
        
        NOTE: We update the buffer even if there is already data from this task.
        By default, this doesn't really change anything, since we keep the same
        amount of data from each class in the buffer.

        Args:
            new_task (Task): New task that is about to be trained on.
            new_task_loader (DataLoader): The dataloader of the corresponding dataset.
        """

        from experiments.addons.replay import LabeledReplayBuffer
        # TODO: For now we assume that the buffer is labeled, for simplicity.
        assert isinstance(self.replay_buffer, LabeledReplayBuffer)
            
        i = new_task.index
        current_size = len(self.replay_buffer)
        n_tasks_in_buffer = len(self.tasks_in_buffer)        
        logger.debug("Updating the replay buffer.")
        logger.debug("Current buffer size: ", current_size)
        logger.debug("number of tasks in the buffer:", n_tasks_in_buffer)

        if new_task not in self.tasks_in_buffer:
            # adding a new task to the buffer.
            space_per_task = self.replay_buffer.capacity // (n_tasks_in_buffer + 1)
        else:
            space_per_task = self.replay_buffer.capacity // n_tasks_in_buffer
        # Add all the new data:
        kept_data = self.choose_samples_to_go_in_buffer(
            samples=itertools.chain(self.replay_buffer, unbatch(new_task_loader)),
            n_samples=self.replay_buffer.capacity,
        )
        self.replay_buffer.extend(kept_data)
        # Count how many samples of each class are in the buffer.
        logger.info(f"# of samples per class in Replay buffer: {self.replay_buffer.samples_per_class()}")
        
    def preprocess(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor], Tuple[Tuple[Tensor,Tensor], Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        if len(batch)==1:
            if isinstance(batch[0], tuple) or isinstance(batch[0], list):
                batch = (torch.stack(batch[0], dim=1),)#.transpose(1,0),)
        elif isinstance(batch[0], tuple) or isinstance(batch[0], list):
            batch[0] = torch.stack(batch[0], dim=1)#.transpose(1,0)
        data, target = super().preprocess(batch)
        #use simclr augmentation
        #if self.config.simclr_augment and self.model.training: 
        #    from tasks.simclr.simclr_task_ptl import SimCLRTrainDataTransform_
        #    from torchvision.transforms import Compose, ToPILImage
        #    transform = Compose([ToPILImage(), SimCLRTrainDataTransform_(dobble=False, input_height=data.shape[-1])])
        #    dev = data.device
        #    data = torch.stack([torch.stack(transform(x_i)) for x_i in data.cpu()], dim=0)  # [2*B, C, H, W]    
        #    data = data.view((-1, *data.shape[2:])).to(dev)
        #    target = torch.cat([target,target], dim=0)
        return data, target
        
    # @property
    # def checkpoints_dir(self) -> Path:
    #     return self.config.log_dir / ("checkpoints"+self.md5)
    
    # @property
    # def md5(self):
    #     from tasks.byol_task import BYOL_Task
    #     return hashlib.md5(str(self).encode('utf-8')+str(Classifier.HParams).encode('utf-8')+str(BYOL_Task.Options).encode('utf-8')).hexdigest()


    def choose_samples_to_go_in_buffer(self, samples: Iterable[Tuple[Tensor, Tensor]],
                                             n_samples: int) -> Iterable[Tuple[Tensor, Tensor]]:
        """ Choose which `n_samples` from `samples` to store in the buffer.
        
        By default, just takes the first `n_samples/n_classes` examples for each
        task, but we could easily implement some K-Means thing here if we wanted.

        TODO: Do we assume that the samples are already preprocessed ? or no?

        Args:
            task (Task): Object describing the new task.
            samples (Iterable[Tuple[Tensor, Tensor]]): Samples
            n_samples (int): Number of samples that should be returned.

        Returns:
            Iterable[Tuple[Tensor, Tensor]]: [description]
        """
        # We do the same for the new data.
        samples_per_class: Dict[Optional[int], List[Tuple[Tensor, Tensor]]] = defaultdict(list)
        
        for x, y in samples:
            label = int(y)
            samples_per_class[label].append((x, y))

        # Round robin: cycle through the labels, taking one element at a time.
        alternate_between_classes = roundrobin(*samples_per_class.values())
        # slice the iterator, returning only the `n_samples` first samples.
        return itertools.islice(alternate_between_classes, n_samples)

    def log(self, message: Dict[str, Any], step: int=None, once: bool=False, prefix: str=""):
        #if not once:
            # We add the currently learned task to the logged message, if logging
            # periodically. (`once` is not True)
        #    message.setdefault("task/currently_learned_task", self.state.i)
        super().log(message, step=step, once=once, prefix=prefix)
    
    def step(self, global_step:int, **kwargs):
        return super().step(global_step, epoch=self.state.epoch, epoch_length=self.epoch_length, update_number=self.batch_idx, **kwargs)
    
    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.batch_idx +=1 
        self.model.train()                    
        for batch in dataloader: 
            data, target = self.preprocess(batch)
            yield self.train_batch(data, target)

    def create_tasks_for_dataset(self, dataset: DatasetConfig) -> List[Task]:
        tasks: List[Task] = []

        # Create the tasks, if they aren't given.
        classes = list(range(dataset.num_classes))
        if self.config.random_class_ordering:
            shuffle(classes)        
        for i, label_group in enumerate(n_consecutive(classes, self.config.n_classes_per_task)):
            task = Task(index=i, classes=sorted(label_group))
            tasks.append(task)
        
        return tasks

    def load_pretrain_dataset(self):
        if self.config.unsupervised_epochs_pretraining>0:
                    pretrain_train_dataset, _ = self.config.pretraining_dataset.load(data_dir=self.config.data_dir)
                    self.pretrain_train_dataset, self.pretrain_valid_dataset = train_valid_split(pretrain_train_dataset, 0.2)
    
    def test(self, dataloader: Union[Dataset, DataLoader], description: str = None, name: str = "Test") -> LossInfo:
        loss_info = super().test(dataloader, description=description, name=name)

        #measure the AUC
        loss_info = self.state.perf_meter.update(loss_info).detach()

        #log validation_full performance
        if loss_info.name=='Valid_full':
            self.log({'Valid_full':loss_info.to_dict()})
        return loss_info 
    
    def set_optimizer_lr(self, lr=0.001):
        for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr

    def load_task_datasets(self, tasks: List[Task]):
        """Create the train, valid, test, and cumulative valid & test datasets
        for each task.
        """
        #self.pretrain_train_dataset = None
        #self.pretrain_valid_dataset = None
        # download the dataset.
        
        train_dataset, valid_dataset, test_dataset = self.load_datasets(train_transform =None,valid_transform =None,test_transform = n_consecutive)

        #these will be applied on the level of each task's dataset
        train_transform = ToTensor()
        test_transform = ToTensor()
        valid_transform = ToTensor()
        if self.config.simclr_augment_train: 
            train_transform = SimCLRTrainDataTransform_(dobble=1, input_height=self.config.dataset.x_shape[-1])
        if self.config.simclr_augment_test:
            test_transform = SimCLREvalDataTransform_(dobble=1, input_height=self.config.dataset.x_shape[-1])
            valid_transform = SimCLREvalDataTransform_(dobble=1, input_height=self.config.dataset.x_shape[-1])
    
                
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

            self.train_datasets.append(TransformedDataset(train, train_transform))
            self.valid_datasets.append(TransformedDataset(valid, valid_transform))
            self.test_datasets.append(TransformedDataset(test, test_transform))

        # Use itertools.accumulate to do the summation of the datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        self.test_cumul_dataset = list(accumulate(self.test_datasets))
       
    def get_pretrained_model_path(self, model:Classifier)->Path:
        #model is identifies by number of epochs for pretraining, encoder type and aux_tasks coefficients
        name = hashlib.md5(str(model.model_pretrained_unique_attributes() + [self.config.unsupervised_epochs_pretraining,self.config.pretraining_dataset.name]).encode('utf-8')).hexdigest()
        path = self.config.log_dir_root.joinpath(f'pretrained_models')
        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath(f'{name}.pth') 
        return path
    
    def freeze_encoder(self, model:Classifier) -> Classifier:
        logger.info("Freezing encoder")
        for param in model.encoder.parameters():
            param.requires_grad = False
        return model

    def train_epoch(self, epoch, *args, **kwargs):
        self.log({'task/epoch':epoch+( (self.config.supervised_epochs_per_task+self.config.unsupervised_epochs_per_task+self.config.finetune_epochs_per_task) * self.state.i)})
        return super().train_epoch(epoch, *args, **kwargs)

    def init_model(self) -> Classifier:
        model = super().init_model()
        if self.config.unsupervised_epochs_pretraining > 0:
            #try to load the model
            loaded = self.load_weights(self.get_pretrained_model_path(model))
            if loaded:
                if model.hparams.freeze_pretrained_model:
                    model = self.freeze_encoder(model)
                self.config.unsupervised_epochs_pretraining = 0
        else:
            pass
        return model

    def save_images(self, i: int, dataset: VisionDataset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")

    @property
    def started(self) -> bool:
        checkpoint_exists = (self.checkpoints_dir / "state.json").exists()
        return super().started and checkpoint_exists








    #functions for making figures (nort really used now) 
    def make_transfer_grid_figure(self,
                                  knn_losses: List[List[LossInfo]],
                                  task_losses: List[List[LossInfo]],
                                  cumul_losses: List[LossInfo]) -> plt.Figure:
        """TODO: (WIP): Create a table that shows forward and backward transfer. 

        Args:
            knn_losses (List[List[LossInfo]]): [description]
            task_losses (List[List[LossInfo]]): [description]
            cumul_losses (List[LossInfo]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            plt.Figure: [description]
        """
        n_tasks = len(task_losses)
        knn_accuracies = np.zeros([n_tasks, n_tasks])
        classifier_accuracies = np.zeros([n_tasks, n_tasks])
        
        from itertools import zip_longest
        text: List[List[str]] = [[""] * n_tasks] * n_tasks # [N,N]
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()

        for i in range(n_tasks):
            for j in range(n_tasks):

                knn_loss = knn_losses[i][j] 
                knn_acc = knn_loss.metrics["KNN"].accuracy
                knn_accuracies[i][j] = knn_acc
                print(f"KNN accuracy {i} {j}: {knn_acc:.3%}", id(knn_loss))
                if j <= i:
                    task_loss = task_losses[i][j]
                    sup_acc = get_supervised_accuracy(task_loss)
                    # print(f"Supervised accuracy {i} {j}: {sup_acc:.3%}")
                else:
                    sup_acc = -1
                classifier_accuracies[i][j] = sup_acc         
        
        np.set_printoptions(precision=10)
        
        fig.suptitle("KNN Accuracies")
        knn_accs = np.array(knn_accuracies)
        logger.info(f"KNN Accuracies: \n{knn_accs}")
        sup_accs = np.array(classifier_accuracies)
        logger.info(f"Supervised Accuracies: \n{sup_accs}")
    
        im = ax.imshow(knn_accs)
        
        # We want to show all ticks...
        ax.set_xticks(np.arange(n_tasks))
        ax.set_yticks(np.arange(n_tasks))
        
        # ... and label them with the respective list entries
        x_labels = [f"Task[{i}]" for i in range(n_tasks)]
        y_labels = [f"Task[{i}]" for i in range(n_tasks)]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        if self.config.use_wandb:
            wandb.log({'knn_accuracies': wandb.plots.HeatMap(x_labels, y_labels, knn_accs, show_text=True)})
            wandb.log({'classifier_accuracies': wandb.plots.HeatMap(x_labels, y_labels, sup_accs, show_text=True)})

        # Loop over data dimensions and create text annotations.
        for i in range(n_tasks):
            for j in range(n_tasks):
                text = ax.text(j, i, knn_accs[i, j], ha="center", va="center", color="w")

        ax.set_title("KNN Accuracies")
        fig.tight_layout()
        # if self.config.debug:
        #     fig.show()

        return fig

    def make_loss_figure(self,
                    results: Dict,
                    run_task_classes: List[List[List[int]]],
                    run_final_task_accuracies: List[Tensor]) -> plt.Figure:
        raise NotImplementedError("Not sure if I should do it manually or in wandb.")
        n_runs = len(run_task_classes)
        n_tasks = len(run_task_classes[0])
        
        fig: plt.Figure = plt.figure()
        
        # Create the title for the figure.
        dataset_name = type(self.dataset).__name__
        figure_title: str = " - ".join(filter(None, [
            self.config.run_group,
            self.config.run_name,
            dataset_name,
            ("(debug)" if self.config.debug else None)
        ]))
        fig.suptitle(figure_title)

        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.set_title("Cumulative Validation Accuracy")
        ax1.set_xlabel("Task ID")
        ax1.set_ylabel("Classification Accuracy")

        cumul_metrics = results["metrics"]
        for metric_name, metrics in cumul_metrics.items():
            if "accuracy" in metrics:
                # stack the accuracies for each run, and use the mean and std for the errorbar plot.
                # TODO: might want to implement the "95% confidence with 1000 bootstraps/etc." from the OML paper. 
                accuracy = torch.stack([torch.as_tensor(run_acc) for run_acc in metrics["accuracy"]])
                accuracy_np = accuracy.detach().numpy()
                accuracy_mean = accuracy_np.mean(axis=0)
                accuracy_std  = accuracy_np.std(axis=0)

                ax1.errorbar(x=np.arange(n_tasks), y=accuracy_mean, yerr=accuracy_std, label=metric_name)
                ax1.set_ylim(bottom=0, top=1)
            elif "l2" in metrics:
                pass # TODO: Maybe plot the MSE (called l2 here) for the auxiliary tasks that aren't doing classification.
        ax1.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc="lower left")

         # stack the lists of tensors into a single total_loss tensor.
        final_task_accuracy = torch.stack(run_final_task_accuracies).detach().numpy()
        task_accuracy_means = np.mean(final_task_accuracy, axis=0)
        task_accuracy_std =   np.std(final_task_accuracy, axis=0)

        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        rects = ax2.bar(x=np.arange(n_tasks), height=task_accuracy_means, yerr=task_accuracy_std)
        from utils.plotting import autolabel
        autolabel(ax2, rects)

        ax2.set_title(f"Final mean accuracy per Task")
        ax2.set_xlabel("Task ID")
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        ax2.set_ylim(bottom=0, top=1)
        
        return fig



if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(TaskIncremental, dest="experiment")
    parser.add_arguments(DatasetConfig, dest="dataset_config")

    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    experiment.launch()
