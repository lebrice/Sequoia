from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict, InitVar, fields
from itertools import accumulate
from pathlib import Path
from random import shuffle
from typing import Dict, Iterable, List, Tuple, Union, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from simple_parsing import choice, field, subparsers
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from common.losses import LossInfo, TrainValidLosses
from common.metrics import Metrics, ClassificationMetrics, RegressionMetrics
from config import Config
from datasets.subset import VisionDatasetSubset
from datasets import DatasetConfig
from experiment import Experiment, ExperimentStateBase
from utils.utils import n_consecutive, rgetattr, rsetattr, common_fields
from utils.json_utils import try_load
from utils import utils
from tasks import Tasks
from sys import getsizeof

from utils.json_utils import JsonSerializable
from simple_parsing import mutable_field, list_field


@dataclass
class State(ExperimentStateBase):
    """Object that contains all the state we want to be able to save/restore.

    We aren't going to parse these from the command-line.
    """
    tasks: List[List[int]] = list_field()

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
    # Cumulative losses after each task
    cumul_losses: List[Optional[LossInfo]] = list_field()
    # Container for train/valid losses that are logged periodically.
    all_losses: TrainValidLosses = mutable_field(TrainValidLosses)


@dataclass
class TaskIncremental(Experiment):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    # Number of classes per task.
    n_classes_per_task: int = 2
    # Wether to sort out the classes in the class_incremental setting.
    random_class_ordering: bool = True
    # Maximum number of epochs of self-supervised training to perform on the
    # task data before switching to supervised training.
    unsupervised_epochs_per_task: int = 5
    # Maximum number of epochs of supervised training to perform on each task's
    # dataset.
    supervised_epochs_per_task: int = 1
    # Wether or not we want to cheat and get access to the task-label at train 
    # and test time. NOTE: This should ideally just be a temporary measure while
    # we try to prove that Self-Supervision can help.
    multihead: bool = False 

    # Path to restore the state from at the start of training.
    # NOTE: Currently, should point to a json file, with the same format as the one created by the `save()` method.
    restore_from_path: Optional[Path] = None

    ###
    ##  Fields that contain the state (not to be parsed from the command-line)
    ## TODO: Figure out a neater way to separate the two than with init=False.
    ###
    state: State = mutable_field(State, init=False)


    def __post_init__(self):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__()
        # The entire training and validation datasets.
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []
        self.state = State()

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
        self.model = self.init_model()
        
        if self.started or self.restore_from_path:
            self.logger.info(f"Experiment was already started in the past.")
            self.restore_from_path = self.checkpoints_dir / "state.json"
            self.logger.info(f"Will load state from {self.restore_from_path}")
            self.load_state(self.restore_from_path)

        if self.done:
            self.logger.info(f"Experiment is already done.")
            # exit()

        if self.state.global_step == 0:
            self.logger.info("Starting from scratch!")
            self.state.tasks = self.create_tasks_for_dataset(self.dataset)
        
        self.tasks = self.state.tasks
        self.save()

        # Load the datasets
        self.load_datasets(self.tasks)
        self.n_tasks = len(self.tasks)

        self.logger.info(f"Class Ordering: {self.state.tasks}")

        if self.state.global_step == 0:
            self.state.knn_losses   = [[None] * self.n_tasks] * self.n_tasks # [N,N]
            self.state.task_losses  = [[None] * (i+1) for i in range(self.n_tasks)] # [N,J]
            self.state.cumul_losses = [None] * self.n_tasks # [N]
        
        for i in range(self.state.i, self.n_tasks):
            self.state.i = i
            self.logger.info(f"Starting task {i} with classes {self.tasks[i]}")
 
            # If we are using a multihead model, we give it the task label (so
            # that it can spawn / reuse the output head for the given task).
            if self.multihead:
                self.model.current_task_id = i

            # Training and validation datasets for task i.
            train_i = self.train_datasets[i]
            valid_i = self.valid_datasets[i]

            with self.plot_region_name(f"Learn Task {i}"):
                # We only train (unsupervised) if there is at least one enabled
                # auxiliary task and if the maximum number of unsupervised
                # epochs per task is greater than zero.
                self_supervision_on = any(task.enabled for task in self.model.tasks.values())

                if self_supervision_on and self.unsupervised_epochs_per_task:
                    # Temporarily remove the labels.
                    with train_i.without_labels(), valid_i.without_labels():
                        # Un/self-supervised training on task i.
                        self.state.all_losses += self.train_until_convergence(
                            train_i,
                            valid_i,
                            max_epochs=self.unsupervised_epochs_per_task,
                            description=f"Task {i} (Unsupervised)",
                        )

                # Train (supervised) on task i.
                self.state.all_losses += self.train_until_convergence(
                    train_i,
                    valid_i,
                    max_epochs=self.supervised_epochs_per_task,
                    description=f"Task {i} (Supervised)",
                )
                self.logger.debug(f"Size the state object: {getsizeof(self.state)}")

            # TODO: save the state during training.
            self.save()
            #  Evaluate on all tasks (as described above).
            cumul_loss = LossInfo(f"cumul_losses[{i}]")
            
            for j in range(self.state.j, self.n_tasks):
                self.state.j = j
                train_j = self.train_datasets[j]
                valid_j = self.valid_datasets[j]

                # Measure how linearly separable the representations of task j
                # are by training and evaluating a KNNClassifier on the data of task j.
                train_knn_loss, valid_knn_loss = self.test_knn(train_j, valid_j, description=f"KNN[{i}][{j}]")
                self.log({
                    f"knn_losses[{i}][{j}]/train": train_knn_loss.to_log_dict(),
                    f"knn_losses[{i}][{j}]/valid": valid_knn_loss.to_log_dict(),
                })
                self.state.knn_losses[i][j] = valid_knn_loss
                accuracy = valid_knn_loss.metrics["KNN"].accuracy
                loss = valid_knn_loss.total_loss

                self.logger.info(f"knn_losses[{i}][{j}]/valid Accuracy: {accuracy:.2%}, loss: {loss}")

                if j <= i:
                    # If we have previously trained on this task:
                    self.model.current_task_id = j
                    loss_j = self.test(dataset=valid_j, description=f"task_losses[{i}][{j}]")
                    cumul_loss += loss_j
                    
                    self.state.task_losses[i][j] = loss_j
                    self.log({f"task_losses[{i}][{j}]": loss_j.to_log_dict()})

                self.save()
            self.state.cumul_losses[i] = cumul_loss
            self.state.j = 0

            valid_log_dict = cumul_loss.to_log_dict()
            self.log({f"cumul_losses[{i}]": valid_log_dict})

        # TODO: Save the results to a json file.
        self.save(self.results_dir)
        # TODO: save the rest of the state.
        
        from utils.plotting import maximize_figure
        # Make the forward-backward transfer grid figure.
        grid = self.make_transfer_grid_figure(self.state.knn_losses, self.state.task_losses, self.state.cumul_losses)
        grid.savefig(self.plots_dir / "transfer_grid.png")
        
        # make the plot of the losses (might not be useful, since we could also just do it in wandb).
        fig = self.make_loss_figure(self.all_losses, self.plot_sections)
        fig.savefig(self.plots_dir / "losses.png")
        
        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(10)

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
        knn_accuracies: List[List[Optional[float]]] = [[None] * n_tasks] * n_tasks # [N,N]
        classifier_accuracies: List[List[Optional[float]]] = [[None] * n_tasks] * n_tasks # [N,N]
        
        from itertools import zip_longest
        text: List[List[str]] = np.zeros(n_tasks, n_tasks, dtype=np.string)

        for i, rows in enumerate(zip(classifier_accuracies, knn_accuracies)):
            knn_accuracies.append([])
            classifier_accuracies.append([])
            for j, (knn_loss, classifier_loss) in enumerate(zip_longest(*rows)):
                knn_acc = knn_loss.metrics["KNN"].accuracy
                text[i][j] = f"KNN Acc: {knn_acc:.2%}"
                if classifier_loss:
                    cls_acc = classifier_loss.metrics["supervised"].accuracy
                    text[i][j] += f"\nAcc: {cls_acc:.2%}"
        # fig: plt.Figure = plt.figure()
        # ax = fig.subplots(1,1)
        # ax.
        print(text)
        plt.table(text)
        raise NotImplementedError("Not sure if I should do it manually or in wandb.")

    
    def load_state(self, state_json_path: Path=None) -> None:
        # TODO: save/restore the state from a previous run.
        # TODO: Work-In-Progress, this is ugly. There is for sure a more elegant
        # way to do this.
        self.logger.info(f"Restoring state from {state_json_path}")        

        if not state_json_path:
            state_json_path = self.checkpoints_dir / "state.json"

        # Load the 'State' object from json
        self.state = State.load_json(state_json_path)

        # If any attributes are common to both the Experiment and the State,
        # copy them over to the Experiment.
        for name, (v1, v2) in common_fields(self, self.state):
            self.logger.info(f"Loaded the {field.name} attribute from the 'State' object.")
            setattr(self, name, v2)

        if self.state.model_weights_path:
            self.logger.info(f"Restoring model weights from {self.state.model_weights_path}")
            self.model.load_state_dict(torch.load(self.state.model_weights_path), strict=False)

        # TODO: Fix this so the global_step is nicely loaded/restored.
        self.global_step = self.state.global_step or self.state.all_losses.latest_step()
        self.logger.info(f"Starting at global step = {self.global_step}.")

    def save(self, save_dir: Path=None) -> None:
        if not save_dir:
            save_dir = self.checkpoints_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        save_json_path = save_dir / "state.json"

        if self.model:
            self.state.model_weights_path = save_dir / "model_weights.pth"
            
            torch.save(self.model.state_dict(), self.state.model_weights_path)    
            self.logger.debug(f"Saved model weights to {self.state.model_weights_path}")
        
        
        for name, (v1, v2) in common_fields(self, self.state):
            self.logger.info(f"Saving the {name} attribute into the 'State' object.")
            setattr(self.state, name, v1)
        
        # TODO: Fix this so the global_step is nicely loaded/restored.
        self.state.global_step = self.global_step

        self.state.save_json(save_json_path)
        self.logger.debug(f"Saved state to {save_json_path}")




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

    def create_tasks_for_dataset(self, dataset: DatasetConfig) -> List[List[int]]:
        tasks: List[List[int]] = []

        # Create the tasks, if they aren't given.
        classes = list(range(dataset.y_shape[0]))
        if self.random_class_ordering:
            shuffle(classes)
        
        for label_group in n_consecutive(classes, self.n_classes_per_task):
            tasks.append(list(label_group))
        
        return tasks

    def load_datasets(self, tasks: List[List[int]]) -> List[List[int]]:
        """Create the train, valid and cumulative datasets for each task.
        
        Returns:
            List[List[int]]: The groups of classes for each task.
        """

        # download the dataset.
        super().load_datasets()
        assert self.dataset.train is not None
        assert self.dataset.valid is not None

        # safeguard the entire training dataset.
        train_full_dataset = self.train_dataset
        valid_full_dataset = self.valid_dataset

        self.train_datasets.clear()
        self.valid_datasets.clear()

        for i, task in enumerate(tasks):
            train = VisionDatasetSubset(train_full_dataset, task)
            valid = VisionDatasetSubset(valid_full_dataset, task)
            self.train_datasets.append(train)
            self.valid_datasets.append(valid)

        # Use itertools.accumulate to do the summation of validation datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))

        for i, (train, valid, cumul) in enumerate(zip(self.train_datasets,
                                                      self.valid_datasets,
                                                      self.valid_cumul_datasets)):
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
