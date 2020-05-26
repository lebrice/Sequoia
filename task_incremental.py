import logging
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass, fields
from itertools import accumulate
from pathlib import Path
from random import shuffle
from sys import getsizeof
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from simple_parsing import choice, field, list_field, mutable_field, subparsers
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image

from common.losses import LossInfo, TrainValidLosses
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from config import Config
from datasets import DatasetConfig
from datasets.subset import VisionDatasetSubset
from experiment import Experiment
from tasks import Tasks
from utils import utils
from utils.json_utils import JsonSerializable
from utils.utils import common_fields, n_consecutive, rgetattr, rsetattr
from common.task import Task

logger = logging.getLogger(__file__)


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
        # Cumulative losses after each task
        cumul_losses: List[Optional[LossInfo]] = list_field()

    def __post_init__(self):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__()
        # The entire training and validation datasets.
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []

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
            if not self.restore_from_path:
                self.restore_from_path = self.checkpoints_dir / "state.json"
            self.logger.info(f"Will load state from {self.restore_from_path}")
            self.load_state(self.restore_from_path)

        if self.done:
            self.logger.info(f"Experiment is already done.")
            # exit()

        if self.state.global_step == 0:
            self.logger.info("Starting from scratch!")
            self.state.tasks = self.create_tasks_for_dataset(self.dataset)
        else:
            self.logger.info(f"Starting from global step {self.state.global_step}")
            self.logger.info(f"i={self.state.i}, j={self.state.j}")
        
        self.tasks = self.state.tasks
        self.save(save_model_weights=False)
        
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
                self.on_task_switch(self.tasks[i])

            # Training and validation datasets for task i.
            train_i = self.train_datasets[i]
            valid_i = self.valid_datasets[i]
            if self.state.j == 0:
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
                    # TODO: save the state during training?.
                    self.state.all_losses += self.train_until_convergence(
                        train_i,
                        valid_i,
                        max_epochs=self.supervised_epochs_per_task,
                        description=f"Task {i} (Supervised)",
                    )
            
            # Save to the 'checkpoints' dir
            self.save()
            
            # Evaluation loop:
            for j in range(self.state.j, self.n_tasks):
                if j == 0:
                    #  Evaluate on all tasks (as described above).
                    if self.state.cumul_losses[i] is not None:
                        logger.warning(
                            f"Cumul loss at index {i} should have been None "
                            f"but is {self.state.cumul_losses[i]}.\n"
                            f"This value be overwritten."
                        )
                    self.state.cumul_losses[i] = LossInfo(f"cumul_losses[{i}]")

                self.state.j = j
                train_j = self.train_datasets[j]
                valid_j = self.valid_datasets[j]

                # Measure how linearly separable the representations of task j
                # are by training and evaluating a KNNClassifier on the data of task j.
                train_knn_loss, valid_knn_loss = self.test_knn(
                    train_j,
                    valid_j,
                    description=f"KNN[{i}][{j}]"
                )
                self.log({
                    f"knn_losses/train/[{i}][{j}]": train_knn_loss,
                    f"knn_losses/valid/[{i}][{j}]": valid_knn_loss,
                })
                self.state.knn_losses[i][j] = valid_knn_loss

                accuracy = valid_knn_loss.metrics["KNN"].accuracy
                self.logger.info(f"knn_losses/valid/[{i}][{j}] Accuracy: {accuracy:.2%}")

                if j <= i:
                    # If we have previously trained on this task:
                    if self.multihead:
                        self.on_task_switch(self.tasks[j])

                    loss_j = self.test(dataset=valid_j, description=f"task_losses[{i}][{j}]")
                    self.state.cumul_losses[i] += loss_j
                    self.state.task_losses[i][j] = loss_j

                    self.log({f"task_losses/[{i}][{j}]": loss_j})

            # Save the state with the new metrics, but no need to save the
            # model weights, as they didn't change.
            self.save(save_model_weights=False)
            
            self.state.j = 0
            cumul_loss = self.state.cumul_losses[i]
            self.log({f"cumul_losses[{i}]": cumul_loss})

        # mark that we're done so we get right back here if we resume a
        # finished experiment
        self.state.i = self.n_tasks
        self.state.j = self.n_tasks

        self.save(self.results_dir) # Save to the 'results' dir.
        
        for i, cumul_loss in enumerate(self.state.cumul_losses):
            assert cumul_loss is not None, f"cumul loss at {i} should not be None!"
            cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
            self.logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
            if self.config.use_wandb:
                wandb.run.summary[f"Cumul Accuracy [{i}]"] = cumul_valid_accuracy

        from utils.plotting import maximize_figure
        # Make the forward-backward transfer grid figure.
        grid = self.make_transfer_grid_figure(
            knn_losses=self.state.knn_losses,
            task_losses=self.state.task_losses,
            cumul_losses=self.state.cumul_losses
        )
        grid.savefig(self.plots_dir / "transfer_grid.png")
        
        # if self.config.debug:
        #     grid.waitforbuttonpress(10)

        # make the plot of the losses (might not be useful, since we could also just do it in wandb).
        # fig = self.make_loss_figure(self.all_losses, self.plot_sections)
        # fig.savefig(self.plots_dir / "losses.png")
        
        # if self.config.debug:
        #     fig.show()
        #     fig.waitforbuttonpress(10)

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
                
                if j <= i:
                    task_loss = task_losses[i][j]
                    sup_acc = task_loss.losses["supervised"].metrics["supervised"].accuracy
                    print(f"Supervised accuracy {i} {j}: {sup_acc:.3%}")
                else:
                    sup_acc = np.nan
                classifier_accuracies[i][j] = sup_acc         
        
        fig.suptitle("KNN Accuracies")
        knn_accs = np.array(knn_accuracies).round(2)
        logger.info(f"KNN Accuracies: \n{knn_accs}")
        sup_accs = np.array(classifier_accuracies).round(2)
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
    
    def load_state(self, state_json_path: Path=None) -> None:
        """ save/restore the state from a previous run. """
        # way to do this.
        self.logger.info(f"Restoring state from {state_json_path}")        

        if not state_json_path:
            state_json_path = self.checkpoints_dir / "state.json"

        # Load the 'State' object from json
        self.state = self.State.load_json(state_json_path)

        # If any attributes are common to both the Experiment and the State,
        # copy them over to the Experiment.
        for name, (v1, v2) in common_fields(self, self.state):
            self.logger.info(f"Loaded the {field.name} attribute from the 'State' object.")
            setattr(self, name, v2)

        if self.state.model_weights_path:
            self.logger.info(f"Restoring model weights from {self.state.model_weights_path}")
            state_dict = torch.load(
                self.state.model_weights_path,
                map_location=self.config.device,
            )
            self.model.load_state_dict(state_dict, strict=False)

        # TODO: Fix this so the global_step is nicely loaded/restored.
        self.global_step = self.state.global_step or self.state.all_losses.latest_step()
        self.logger.info(f"Starting at global step = {self.global_step}.")

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

    def create_tasks_for_dataset(self, dataset: DatasetConfig) -> List[Task]:
        tasks: List[Task] = []

        # Create the tasks, if they aren't given.
        classes = list(range(dataset.y_shape[0]))
        if self.random_class_ordering:
            shuffle(classes)
        
        for i, label_group in enumerate(n_consecutive(classes, self.n_classes_per_task)):
            task = Task(index=i, classes=sorted(label_group))
            tasks.append(task)
        
        return tasks

    def load_datasets(self, tasks: List[Task]) -> None:
        """Create the train, valid and cumulative datasets for each task.
        
        Returns:
            List[List[int]]: The groups of classes for each task.
        """
        # download the dataset.
        self.train_dataset, self.valid_dataset = super().load_datasets()

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
        

    def save_images(self, i: int, dataset: VisionDatasetSubset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")

    def on_task_switch(self, task: Task) -> None:
        if self.multihead:
            self.model.on_task_switch(task)

    @property
    def started(self) -> bool:
        checkpoint_exists = (self.checkpoints_dir / "state.json").exists()
        return super().started and checkpoint_exists


def get_supervised_accuracy(cumul_loss: LossInfo) -> float:
    # TODO: this is ugly. There is probably a cleaner way, but I can't think of it right now. 
    try:
        return cumul_loss.losses["Test"].losses["supervised"].metrics["supervised"].accuracy
    except KeyError as e:
        print(cumul_loss)
        raise e


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
