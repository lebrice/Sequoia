from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from random import shuffle
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from simple_parsing import choice, field, subparsers
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from common.losses import LossInfo
from config import Config
from datasets.dataset import TaskConfig
from datasets.subset import VisionDatasetSubset
from experiments.class_incremental import ClassIncremental
from experiments.experiment import Experiment
from utils.utils import n_consecutive, rgetattr, rsetattr

@dataclass
class TaskIncremental(Experiment):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.

    """
    n_classes_per_task: int = 2      # Number of classes per task.
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = True
    epochs_per_task: int = 1  # Number of epochs on each task's dataset.
    
    # Number of runs to execute in order to create the OML Figure 3.
    n_runs: int = 5

    def __post_init__(self):
        super().__post_init__()
        self.train_full_dataset: VisionDataset = None
        self.valid_full_dataset: VisionDataset = None
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []
        self.task_classes: List[List[int]] = list()

    def run(self):
        valid_losses_list: List[List[LossInfo]] = []
        final_task_accuracies_list: List[Tensor] = []

        for i in range(self.n_runs):
            print(f"STARTING RUN {i}")
            valid_losses, final_task_accuracies = self._run()
            valid_losses_list.append(valid_losses)
            final_task_accuracies_list.append(final_task_accuracies)

        valid_loss = stack_loss_attr(valid_losses_list, "total_loss")
        final_task_accuracy = torch.stack(final_task_accuracies_list)
        loss_means = valid_loss.mean(dim=0).numpy()
        loss_stds = valid_loss.std(dim=0).numpy()
        
        task_accuracy_means = final_task_accuracy.mean(dim=0).numpy()
        task_accuracy_std =   final_task_accuracy.std(dim=0).numpy()
        n_tasks= len(task_accuracy_means)

        print("CUMULATIVE VALID LOSS:")
        print(valid_loss)
        print("FINAL MEAN TASK ACCURACies:")
        print(final_task_accuracy)    

        print("Loss Means:", loss_means)
        print("Loss STDs:", loss_stds)

        print("Final Task Accuracy means:", task_accuracy_means)
        print("Final Task Accuracy stds:", task_accuracy_std)
        
        fig: plt.Figure = plt.figure()

        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.errorbar(x=np.arange(n_tasks), y=loss_means, yerr=loss_stds, label=self.config.run_name)
        ax1.set_title("Continual Classification Accuracy")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Classification Loss")
        ax1.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc="upper left")

        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        for todo in range(1):
            ax2.bar(x=np.arange(n_tasks), height=task_accuracy_means, yerr=task_accuracy_std)
        ax2.set_title(f"Final mean accuracy per Task")
        ax2.set_xlabel("Task ID")
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=10)
        fig.savefig(self.plots_dir / "oml_fig.jpg")
        self.log({"OML": fig}, once=True)

        return 
    
    def _run(self) -> Tuple[List[LossInfo], Tensor]:
        """Executes one single run from the OML figure 3.
        
        Trains the model until convergence on each task, using the validation
        set specific to each task. Then, evaluates the model on the cumulative
        validation set.
        
        This returns a list cumulative validation losses, as well as the final
        average class accuracy for each task. 
        
        Returns:
            Tuple[List[LossInfo], Tensor]: List of total losses on the
            cummulative validation dataset after learning each task, as well as
            a Tensor holding the final average class accuracy for each task.
        """
        label_order = self.load()
        datasets = zip(
            self.train_datasets,
            self.valid_datasets,
            self.valid_cumul_datasets
        )
        
        train_losses: List[LossInfo] = []
        valid_losses: List[LossInfo] = []

        for task_index, (train, valid, valid_cumul) in enumerate(datasets):
            print(f"Starting task {task_index}, Classes {self.task_classes[task_index]}")
            
            # Train on the current task:
            self.train_until_convergence(
                train,
                valid,
                max_epochs=self.epochs_per_task,
                description=f"Task {task_index} ",
            )
            
            ## TODO: turned this off for now, not sure if OML paper does this.
            # # Evaluate the performance on training set.
            # # We only really do this in order to get a plot like the one in OML.
            # train_loss = self.test(train, description=f"Task {task_index} Train")

            # Evaluate the performance on the cumulative validation set.
            valid_loss = self.test(valid_cumul, description=f"Task {task_index} Train ")

            # train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            validation_metrics = valid_loss.metrics
            class_accuracy = validation_metrics.class_accuracy  
            print(f"AFTER TASK {task_index}:",
                  f"\tCumulative Val Loss: {valid_loss.total_loss},",
                  f"\tClass Accuracy: {class_accuracy}", sep=" ")

        task_mean_accuracy = torch.zeros(len(self.task_classes))
        # get the average accuracy per task:
        for i, label_group in enumerate(self.task_classes):
            task_mean_accuracy[i] = class_accuracy[label_group].mean()
        
        # Get the most recent class accuracy metrics.
        return valid_losses, task_mean_accuracy


    def load(self):
        """ Create the train, valid and cumulative datasets for each task. """
        # download the dataset.
        self.dataset.load(self.config)

        assert self.dataset.train is not None
        assert self.dataset.valid is not None

        # safeguard the entire training dataset.
        self.train_full_dataset = self.dataset.train
        self.valid_full_dataset = self.dataset.valid
        self.train_datasets.clear()
        self.valid_datasets.clear()
        self.task_classes.clear()
        all_labels = list(range(self.dataset.y_shape[0]))
        
        if self.random_class_ordering:
            shuffle(all_labels)
        print("Class Ordering:", all_labels)
        
        for label_group in n_consecutive(all_labels, self.n_classes_per_task):
            train = VisionDatasetSubset(self.train_full_dataset, label_group)
            self.train_datasets.append(train)

            valid = VisionDatasetSubset(self.valid_full_dataset, label_group)
            self.valid_datasets.append(valid)

            self.task_classes.append(list(label_group))

        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        
        for i, dataset in enumerate(self.train_datasets):
            self.save_images(i, dataset, prefix="train_")
        
        for i, dataset in enumerate(self.valid_datasets):
            self.save_images(i, dataset, prefix="valid_")
        
        for i, dataset in enumerate(self.valid_cumul_datasets):
            self.save_images(i, dataset, prefix="valid_cumul_")
        
        return all_labels

    def save_images(self, i: int, dataset: VisionDatasetSubset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")
        
    def set_task(self, task_index: int) -> Tuple[DataLoader, DataLoader]:
        assert 0 <= task_index < len(self.train_datasets)
        self.dataset.train = self.train_datasets[task_index]
        self.dataset.valid = self.valid_cumul_datasets[task_index]
        ## equivalent to super().load()
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders
        return self.train_loader, self.valid_loader


def stack_loss_attr(losses: List[List[LossInfo]], attribute: str="total_loss") -> Tensor:
    n = len(losses)
    length = len(losses[0])
    result = torch.zeros([n, length], dtype=torch.float)
    for i, run_losses in enumerate(losses):
        for j, epoch_loss in enumerate(run_losses):
            result[i,j] = rgetattr(epoch_loss, attribute)
    return result
