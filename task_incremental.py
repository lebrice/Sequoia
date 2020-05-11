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
from datasets.subset import VisionDatasetSubset
from experiment import Experiment
from utils.utils import n_consecutive, rgetattr, rsetattr
from utils import utils
from tasks import Tasks

@dataclass
class TaskIncremental(Experiment):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    # Number of classes per task.
    n_classes_per_task: int = 2
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = True
    # (Maximum number of epochs of self-supervised training to perform before switching
    # to supervised training.
    unsupervised_epochs_per_task: int = 5
    # (Maximum number of epochs of supervised training to perform on each task's dataset.
    supervised_epochs_per_task: int = 1
    # Number of runs to execute in order to create the OML Figure 3.
    n_runs: int = 5
    # Wether or not we want to cheat and get access to the task-label at train 
    # and test time. NOTE: This should ideally just be a temporary measure while
    # we try to prove that Self-Supervision can help.
    multihead: bool = False 

    def __post_init__(self):
        super().__post_init__()
        # The entire training and validation datasets.
        self.train_full_dataset: VisionDataset = None
        self.valid_full_dataset: VisionDataset = None
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []

    def run(self):
        # containers for the results of each individual run.
        cumul_valid_losses: List[List[LossInfo]] = []
        # Stores the final mean task accuracy per class.
        final_task_accuracies: List[Tensor] = []
        # stores the list of tasks (groups of labels) used in each run.
        tasks: List[List[List[int]]] = []
        
        for i in range(self.n_runs):
            print(f"STARTING RUN {i}")
            # Set a different random seed for each run.
            utils.set_seed(self.config.random_seed + i)

            # Get the tasks to use for this run.
            run_tasks = self.load()
            # Get the resulting cumulative validation losses.
            run_cumul_valid_losses: List[LossInfo] = self.run_once(run_tasks)

            final_loss = run_cumul_valid_losses[-1]
            run_final_task_accuracies = get_mean_task_accuracy(final_loss, run_tasks)

            # Accumulate the results of this run in the above lists
            tasks.append(run_tasks)
            cumul_valid_losses.append(run_cumul_valid_losses)
            final_task_accuracies.append(run_final_task_accuracies)
            
            # results: Dict = {
            #     "tasks": tasks,
            #     "cumul_valid_losses": cumul_valid_losses,
            #     "final_task_accuracy": torch.stack(final_task_accuracies),
            # }

            # create a "stacked"/"serializable" version of the loss objects.
            results: Dict = make_results_dict(cumul_valid_losses)
            # add the tasks to `results` so we save it in the json file.
            results["task_classes"] = tasks
            # Save after each run, just in case we interrupt anything, so we
            # still get partial results even if something goes wrong at some
            # point.
            self.save_to_results_dir({
                "results.json": results,
                "final_task_accuracy.csv": torch.stack(final_task_accuracies).cpu().numpy().tolist(),
            })

            fig: plt.Figure = self.make_figure(
                results,
                tasks,
                final_task_accuracies,
            )

            if self.config.debug:
                fig.show()
                fig.waitforbuttonpress(timeout=10)
            
            fig.savefig(self.plots_dir / "oml_fig.jpg")
            self.log({"oml_fig.jpg": fig}, once=True)

        task_accuracy = torch.stack(final_task_accuracies)
        task_accuracy_means = task_accuracy.mean(dim=0).detach().numpy()
        task_accuracy_stds  = task_accuracy.std(dim=0).detach().numpy()
        self.log({
            "Final Task Accuracy means:": task_accuracy_means,
            "Final Task Accuracy stds:": task_accuracy_stds,
            }, once=True, always_print=True)

    def make_figure(self,
                    results: Dict,
                    run_task_classes: List[List[List[int]]],
                    run_final_task_accuracies: List[Tensor]) -> plt.Figure:
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

    def run_once(self, tasks: List[List[int]]) -> Tuple[List[LossInfo], List[List[int]]]:
        """Executes one single run from the OML figure 3.
        
        Trains the model until convergence on each task, using the validation
        set specific to each task. Then, evaluates the model on the cumulative
        validation set.

        Args:
            tasks: List[List[int]] A list containing a list of the
            classes learned during each task.
        
        Returns:
            valid_cumul_losses: List[LossInfo] List of total losses on the
            cummulative validation dataset after learning each task
        """
        self.init_model()

        label_order: List[int] = sum(tasks, [])
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
            if self.multihead:
                self.model.current_task_id = task_index
            
            classes = tasks[task_index]
            print(f"Starting task {task_index}, Classes {classes}")

            # if there are any enabled auxiliary tasks:
            if any(task.enabled for task in self.model.tasks.values()):
                # temporarily remove the labels
                with train.without_labels(), valid.without_labels():
                    # Train (Unsupervised/Self-supervised)
                    self.train_until_convergence(
                        train,
                        valid,
                        max_epochs=self.unsupervised_epochs_per_task,
                        description=f"Task {task_index} (Unsupervised)",
                    )
            # Train (supervised)
            self.train_until_convergence(
                train,
                valid,
                max_epochs=self.supervised_epochs_per_task,
                description=f"Task {task_index} (Supervised)",
            )
            
            # Evaluate the performance on the cumulative validation set.
            if not self.multihead:
                # just use the whole cumulative validation set.
                valid_loss = self.test(
                    valid_cumul,
                    description=f"Task {task_index} Valid (Cumul) "
                )
            else:
                # If we're cheating, then use the validation set for each task,
                # and add up the results. This is easier than having to use
                # different classifiers depending on the labels for each sample.
                valid_loss = LossInfo("Test")
                # evaluate from task_id 0 to the current task_id.
                for task_id in range(task_index + 1):
                    self.model.current_task_id = task_id
                    valid_dataset = self.valid_datasets[task_id]
                    valid_loss += self.test(
                        dataset=valid_dataset,
                        description=f"Task {task_index} Valid for Task {task_id} "
                    )

            # train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            # NOTE: This is just a bugfix while I'm refactoring this mess.
            supervised_metrics = valid_loss.losses[Tasks.SUPERVISED].metrics[Tasks.SUPERVISED]
            # print("Supervised metrics", supervised_metrics)
            assert isinstance(supervised_metrics, ClassificationMetrics)
            class_accuracy = supervised_metrics.class_accuracy
            # print(f"AFTER TASK {task_index}:",
            #       f"\tCumulative Val Loss: {valid_loss.total_loss},",
            #       f"\tMean Class Accuracy: {class_accuracy.mean()}", sep=" ")

        return valid_losses

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

        tasks: List[List[int]] = []
        
        for label_group in n_consecutive(all_labels, self.n_classes_per_task):
            train = VisionDatasetSubset(self.train_full_dataset, label_group)
            self.train_datasets.append(train)

            valid = VisionDatasetSubset(self.valid_full_dataset, label_group)
            self.valid_datasets.append(valid)

            tasks.append(list(label_group))

        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        
        datasets = zip(self.train_datasets,
                       self.valid_datasets,
                       self.valid_cumul_datasets)
        for i, (train, valid, cumul) in enumerate(datasets):
            self.save_images(i, train, prefix="train_")
            self.save_images(i, valid, prefix="valid_")
            self.save_images(i, cumul, prefix="valid_cumul_")
        
        return tasks

    def save_images(self, i: int, dataset: VisionDatasetSubset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")


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


def make_results_dict(run_cumul_valid_losses: List[List[LossInfo]]) -> Dict:
    # get a "stacked" version of the loss dicts, so that we get dicts of
    # lists of tensors.
    stacked: Dict[str, Union[Tensor, Dict]] = stack_dicts([
        stack_dicts(losses) for losses in run_cumul_valid_losses 
    ])
    def to_lists(tensors: Union[List, Dict]) -> Union[List, Dict]:
        """ Converts all the tensors within `tensors` to lists."""
        if isinstance(tensors, list) and tensors:
            if isinstance(tensors[0], Tensor):
                return torch.stack(tensors).tolist()
            elif isinstance(tensors[0], list):
                return list(map(to_lists, tensors))
        elif isinstance(tensors, dict):
            for key, values in tensors.items():
                if isinstance(values, (dict, list)):
                    tensors[key] = to_lists(values)
                elif isinstance(values, Tensor):
                    tensors[key] = values.tolist()
        return tensors

    stacked = to_lists(stacked)
    return stacked


def get_mean_task_accuracy(loss: LossInfo, run_tasks: List[List[int]]) -> Tensor:
    """Gets the mean classification accuracy for each task.
    
    Args:
        loss (LossInfo): A given LossInfo. (usually the last of the cumulative
        validation losses).
        run_tasks (List[List[int]]): The classes within each task.
    
    Returns:
        Tensor: Float tensor of shape [len(run_tasks)] containing the mean
        accuracy for each task. 
    """
    # get the last validation metrics.
    metrics = loss.losses[Tasks.SUPERVISED].metrics
    classification_metrics: ClassificationMetrics = metrics[Tasks.SUPERVISED]  # type: ignore
    final_class_accuracy = classification_metrics.class_accuracy

    # Find the mean accuracy per task at the end of training.
    final_accuracy_per_task = torch.zeros(len(run_tasks))
    for task_index, classes in enumerate(run_tasks):
        task_class_accuracies = final_class_accuracy[classes]
        final_accuracy_per_task[task_index] = task_class_accuracies.mean()
    return final_accuracy_per_task


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TaskIncremental, dest="experiment")
    
    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    
    from main import launch
    launch(experiment)
