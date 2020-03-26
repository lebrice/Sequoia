from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
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
from common.metrics import Metrics, ClassificationMetrics, RegressionMetrics
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

    # (Maximum number of epochs of self-supervised training to perform before switching
    # to supervised training.
    unsupervised_epochs_per_task: int = 5
    # (Maximum number of epochs of supervised training to perform on each task's dataset.
    supervised_epochs_per_task: int = 1

    # Number of runs to execute in order to create the OML Figure 3.
    n_runs: int = 5

    def __post_init__(self):
        super().__post_init__()
        self.train_full_dataset: VisionDataset = None
        self.valid_full_dataset: VisionDataset = None
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []

    def run(self) -> Dict[Path, Tensor]:
        valid_losses_list: List[List[LossInfo]] = []
        final_task_accuracies_list: List[Tensor] = []
        results: Dict[Path, Tensor] = {}

        for i in range(self.n_runs):
            print(f"STARTING RUN {i}")
            # execute a single run
            cumul_valid_losses, task_classes = self._run()

            final_task_accuracies = get_mean_task_accuracy_at_end_of_training(
                cumul_valid_losses,
                task_classes
            )

            valid_losses_list.append(cumul_valid_losses)
            final_task_accuracies_list.append(final_task_accuracies)

            # stack the lists of tensors into a single total_loss tensor.
            # TODO: save everything useful in LossInfo, not just the total_loss.
            total_valid_loss = stack_loss_attr(valid_losses_list, "total_loss")
            final_task_accuracy = torch.stack(final_task_accuracies_list)

            # Save after each run, just in case we interrupt anything, so we
            # still get partial results even if something goes wrong at some
            # point.
            results["valid_loss.csv"] = total_valid_loss
            results["final_task_accuracy.csv"] = final_task_accuracy
            self.save_results(results)
            self.make_figure(
                valid_losses_list,
                final_task_accuracies_list,
            )
        
        loss_means = total_valid_loss.mean(dim=0).detach().numpy()
        loss_stds = total_valid_loss.std(dim=0).detach().numpy()
        task_accuracy_means = final_task_accuracy.mean(dim=0).detach().numpy()
        task_accuracy_std =   final_task_accuracy.std(dim=0).detach().numpy()
            
        n_tasks= len(task_accuracy_means)

        self.log({
            "Cumulative valid loss": total_valid_loss,
            "Final mean task accuracies": final_task_accuracy,
            "Loss Means": loss_means,
            "Loss STDs": loss_stds,
            "Final Task Accuracy means:": task_accuracy_means,
            "Final Task Accuracy stds:": task_accuracy_std,
            }, once=True, always_print=True)
    
    def make_figure(self,
                    cumul_valid_losses: List[List[LossInfo]],
                    final_task_accuracies: List[Tensor]):
        n_runs = len(cumul_valid_losses)
        # stack the lists of tensors into a single total_loss tensor.
        final_task_accuracy = torch.stack(final_task_accuracies).detach().numpy()
        task_accuracy_means = np.mean(final_task_accuracy, axis=0)
        task_accuracy_std =   np.std(final_task_accuracy, axis=0)

        # get a "stacked" version of the metrics dicts, so that we get dicts of
        # lists of tensors.
        stacked: Dict[str, Union[Tensor, Dict]] = stack_dicts([
            stack_dicts(losses) for losses in cumul_valid_losses 
        ])

       
        # valid_loss = stack_loss_attr(cumul_valid_losses, "total_loss")
        # cumul_metrics = arrange_metrics_by_name(cumul_valid_losses)
        valid_loss = stacked["total_loss"]
        cumul_metrics = stacked["metrics"]
                
        loss_stds = np.std(valid_loss, axis=0)
        
        n_tasks= len(task_accuracy_means)

        dataset_name = type(self.dataset).__name__

        fig: plt.Figure = plt.figure()
        fig.suptitle(f"{self.config.run_group} - {self.config.run_name} - {dataset_name}")
        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.set_title("Cumulative Validation Accuracy")
        ax1.set_xlabel("Number of tasks learned")
        ax1.set_ylabel("Classification Accuracy")
        for metric_name, metrics in cumul_metrics.items():
            if "accuracy" in metrics:
                accuracy = torch.stack([torch.Tensor(run_acc) for run_acc in metrics["accuracy"]])
                accuracy_mean = accuracy.mean(dim=0).detach().numpy()
                accuracy_std = accuracy.std(dim=0).detach().numpy()
                ax1.errorbar(x=np.arange(n_tasks), y=accuracy_mean, yerr=accuracy_std, label=metric_name)
        ax1.set_xticks(np.arange(n_tasks, dtype=int))
        ax1.legend(loc="lower left")

        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        ax2.bar(x=np.arange(n_tasks), height=task_accuracy_means, yerr=task_accuracy_std)
        ax2.set_title(f"Final mean accuracy per Task")
        ax2.set_xlabel("Task ID")
        ax2.set_xticks(np.arange(n_tasks, dtype=int))
        
        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=10)
        
        fig.savefig(self.plots_dir / "oml_fig.jpg")
        self.log({"oml_fig.jpg": fig}, once=True)


    def _run(self) -> Tuple[List[LossInfo], List[List[int]]]:
        """Executes one single run from the OML figure 3.
        
        Trains the model until convergence on each task, using the validation
        set specific to each task. Then, evaluates the model on the cumulative
        validation set.
        
        Returns:
            valid_cumul_losses: List[LossInfo] List of total losses on the
            cummulative validation dataset after learning each task

            task_classes: List[List[int]] A list containing a list of the
            classes learned during each task.
        """
        task_classes = self.load()
        label_order = sum(task_classes, [])
        print("Class Ordering:", label_order)
        datasets = zip(
            self.train_datasets,
            self.valid_datasets,
            self.valid_cumul_datasets
        )
        
        train_losses: List[LossInfo] = []
        valid_losses: List[LossInfo] = []

        train: VisionDatasetSubset
        valid: VisionDatasetSubset
        valid_cumul: VisionDatasetSubset
        
        for task_index, (train, valid, valid_cumul) in enumerate(datasets):
            classes = task_classes[task_index]
            print(f"Starting task {task_index}, Classes {classes}")

            # if there are any enabled auxiliary tasks:
            if any(task.enabled for task in self.model.tasks.values()):
                # temporarily remove the labels
                with train.without_labels(), valid.without_labels():
                    self.train_until_convergence(
                        train,
                        valid,
                        max_epochs=self.unsupervised_epochs_per_task,
                        description=f"Task {task_index} (Unsupervised)",
                    )
            self.train_until_convergence(
                train,
                valid,
                max_epochs=self.supervised_epochs_per_task,
                description=f"Task {task_index} (Supervised)",
            )
            
            # Evaluate the performance on the cumulative validation set.
            valid_loss = self.test(
                valid_cumul,
                description=f"Task {task_index} Valid (Cumul) "
            )

            # train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            validation_metrics: Dict[str, Metrics] = valid_loss.metrics
            class_accuracy = validation_metrics["supervised"].class_accuracy  
            # print(f"AFTER TASK {task_index}:",
            #       f"\tCumulative Val Loss: {valid_loss.total_loss},",
            #       f"\tMean Class Accuracy: {class_accuracy.mean()}", sep=" ")

        return valid_losses, task_classes


    def load(self) -> List[List[int]]:
        """Create the train, valid and cumulative datasets for each task. 
        
        Returns:
            List[List[int]]: The groups of classes for each task.
        """
        # download the dataset.
        self.dataset.load(self.config)

        assert self.dataset.train is not None
        assert self.dataset.valid is not None

        # safeguard the entire training dataset.
        self.train_full_dataset = self.dataset.train
        self.valid_full_dataset = self.dataset.valid
        

        self.train_datasets.clear()
        self.valid_datasets.clear()
        all_labels = list(range(self.dataset.y_shape[0]))
        
        if self.random_class_ordering:
            shuffle(all_labels)

        task_classes: List[List[int]] = []
        
        for label_group in n_consecutive(all_labels, self.n_classes_per_task):
            train = VisionDatasetSubset(self.train_full_dataset, label_group)
            self.train_datasets.append(train)

            valid = VisionDatasetSubset(self.valid_full_dataset, label_group)
            self.valid_datasets.append(valid)

            task_classes.append(list(label_group))

        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        
        datasets = zip(self.train_datasets,
                       self.valid_datasets,
                       self.valid_cumul_datasets)
        for i, (train, valid, cumul) in enumerate(datasets):
            self.save_images(i, train, prefix="train_")
            self.save_images(i, valid, prefix="valid_")
            self.save_images(i, cumul, prefix="valid_cumul_")
        
        return task_classes

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


def stack_loss_attr(losses: List[List[LossInfo]], attribute: str) -> Tensor:
    n = len(losses)
    length = len(losses[0])
    result = torch.zeros([n, length], dtype=torch.float)
    for i, run_losses in enumerate(losses):
        for j, epoch_loss in enumerate(run_losses):
            result[i,j] = rgetattr(epoch_loss, attribute)
    return result


def stack_dicts(values: List[Union[Metrics, LossInfo, Dict]]) -> Dict[str, Union[List, Dict[str, List]]]:
    result: Dict[str, List] = OrderedDict()
    
    # do a pass throught the list, adding the dictionary elements.
    for loss in values:
        if isinstance(loss, dict):
            loss_dict = loss
        elif isinstance(loss, (LossInfo, Metrics)):
            loss_dict = loss.to_log_dict()
        
        for key, value in loss_dict.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    # do a second pass, and recurse if there are non-flattened dicts
    for key, values in result.items():
        if isinstance(values[0], (dict, Metrics, LossInfo)):
            result[key] = stack_dicts(values)
    return result


def arrange_metrics_by_name(losses: List[List[LossInfo]]) -> Dict[str, List[List[Metrics]]]:
    """Create a dict of metrics for each run from the list of `LossInfo`s.
    
    Args:
        losses (List[List[LossInfo]]): For each run, the list of `LossInfo`s.  
    
    Returns:
        Dict[str, List[List[Metrics]]]: A dict of the form {<metric_name>: [[<run_1_metrics>], [<run_2_metrics>], ...]}
    """
    ## TODO: get a nice list of lists of all the metrics, arranged by name.
    from utils.utils import to_dict_of_lists
    metrics: Dict[str, List[List[Metrics]]] = defaultdict(list)

    for run_number, run_losses in enumerate(losses):
        run_metrics_dict = to_dict_of_lists(loss.metrics for loss in run_losses)
        for metric_name, metric_values in run_metrics_dict.items():
            metrics[metric_name].append(metric_values)
    return metrics


def get_mean_task_accuracy_at_end_of_training(cumul_valid_losses: List[LossInfo],
                                                task_classes: List[List[int]]) -> Tensor:
    """Gets the mean accuracy within each task at the end of training.
    
    Args:
        cumul_valid_losses (List[LossInfo]): The list of losses.
        task_classes (List[List[int]]): The classes within each task.
    
    Returns:
        Tensor: Float tensor of shape [len(task_classes), 1] containing the mean accuracy for each task. 
    """
    # get the last validation metrics.
    last_metrics = cumul_valid_losses[-1].metrics
    classification_metrics: ClassificationMetrics = last_metrics["supervised"]
    final_class_accuracy = classification_metrics.class_accuracy

    # Find the mean accuracy per task at the end of training.
    final_accuracy_per_task = torch.zeros(len(task_classes))
    for task_index, classes in enumerate(task_classes):
        task_class_accuracies = final_class_accuracy[classes]
        final_accuracy_per_task[task_index] = task_class_accuracies.mean()
    return final_accuracy_per_task
