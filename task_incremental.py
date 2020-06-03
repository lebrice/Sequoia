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
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from common.losses import LossInfo, TrainValidLosses
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from config import Config
from datasets import DatasetConfig
from datasets.subset import VisionDatasetSubset
from torchvision.datasets import VisionDataset
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

        # Knn Losses on the entire test set after having learned each task.
        knn_full_losses: List[LossInfo] = list_field()

        # Cumulative losses after each task
        cumul_losses: List[Optional[LossInfo]] = list_field()

    def __post_init__(self):
        """ NOTE: fields that are created in __post_init__ aren't serialized to/from json! """
        super().__post_init__()


        # The entire training, validation and testing datasets.
        self.full_train_dataset : VisionDataset = None
        self.full_valid_dataset : VisionDataset = None
        self.full_test_dataset  : VisionDataset = None
        
        # Datasets for each task
        self.train_datasets: List[VisionDatasetSubset] = []
        self.valid_datasets: List[VisionDatasetSubset] = []
        self.test_datasets: List[VisionDatasetSubset] = []
        
        # Cumulative datasets: Hold the data from previously seen tasks
        self.valid_cumul_datasets: List[VisionDatasetSubset] = []
        self.test_cumul_datasets: List[VisionDatasetSubset] = []

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
            logger.info(f"Experiment was already started in the past.")
            if not self.restore_from_path:
                self.restore_from_path = self.checkpoints_dir / "state.json"
            logger.info(f"Will load state from {self.restore_from_path}")
            self.load_state(self.restore_from_path)


        if self.done:
            logger.info(f"Experiment is already done.")
            # exit()

        if self.state.global_step == 0:
            logger.info("Starting from scratch!")
            self.state.tasks = self.create_tasks_for_dataset(self.dataset)
        else:
            logger.info(f"Starting from global step {self.state.global_step}")
            logger.info(f"i={self.state.i}, j={self.state.j}")
        
        self.tasks = self.state.tasks
        self.save_state(save_model_weights=False)
        
        # Load the datasets
        self.load_task_datasets(self.tasks)
        self.n_tasks = len(self.tasks)

        logger.info(f"Class Ordering: {self.state.tasks}")
        
        if self.state.global_step == 0:
            self.state.knn_losses   = [[None for _ in range(self.n_tasks)] for _ in range(self.n_tasks)]
            self.state.knn_full_losses   = [None for _ in range(self.n_tasks)]
            self.state.task_losses  = [[None for _ in range(i+1)] for i in range(self.n_tasks)] # [N,J]
            self.state.cumul_losses = [None for _ in range(self.n_tasks)] # [N]

        for i in range(self.state.i, self.n_tasks):
            self.state.i = i
            logger.info(f"Starting task {i} with classes {self.tasks[i]}")

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
                            self.state.all_losses = self.train(
                                train_i,
                                valid_i,
                                epochs=self.unsupervised_epochs_per_task,
                                description=f"Task {i} (Unsupervised)",
                                use_accuracy_as_metric=False, # Can't use accuracy as metric during unsupervised training.
                                temp_save_dir=self.checkpoints_dir / f"task_{i}_unsupervised",
                            )

                    # Train (supervised) on task i.
                    # TODO: save the state during training?.
                    self.state.all_losses += self.train(
                        train_i,
                        valid_i,
                        epochs=self.supervised_epochs_per_task,
                        description=f"Task {i} (Supervised)",
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

                # -- Evaluate Representations after having learned tasks [0:i] on data from task J. --

                train_j = self.train_datasets[j]
                test_j  = self.test_datasets[j]
                # Measure the "quality" of the representations, by training and
                # evaluating a classifier on train and test data from task J.
                knn_j_train_loss, knn_j_test_loss = self.test_knn(
                    train_j,
                    test_j,
                    description=f"KNN [{i}][{j}]"
                )

                knn_j_train_acc = knn_j_train_loss.metric.accuracy
                knn_j_test_acc = knn_j_test_loss.metric.accuracy
                logger.info(f"Task{i}: KNN Train Accuracy [{j}]: {knn_j_train_acc:.2%}")
                logger.info(f"Task{i}: KNN Test  Accuracy [{j}]: {knn_j_test_acc :.2%}")
                # Log the accuracies to wandb.
                self.log({
                    f"KNN/train/task{j}": knn_j_train_acc,
                    f"KNN/test/task{j}" : knn_j_test_acc,
                })
                self.state.knn_losses[i][j] = knn_j_test_loss

                if j <= i:
                    # If we have previously trained on this task:
                    if self.multihead:
                        self.on_task_switch(self.tasks[j])

                    # Test on the test dataset for task j.
                    loss_j = self.test(test_j, description=f"Task{i}: Test on Task{j}", name=f"Task{j}")
                    self.log({f"Task_losses/Task{j}": loss_j})
                    
                    self.state.task_losses[i][j] = loss_j
                    self.state.cumul_losses[i].absorb(loss_j) 
                    # NOTE: using += above would add a "Task<j>" item in the
                    # `losses` attribute of the cumulative loss, without merging the metrics.
                    logger.info(f"Task {i} Supervised Test accuracy on task {j}: ")
                    logger.debug(f"self.state.cumul_losses[i]: {self.state.cumul_losses[i]}")
            
            # -- Evaluate representations after task i on the whole train/test datasets. --

            # Measure the "quality" of the representations of the data using the
            # whole test dataset.
            # TODO: Do this in another process (as it should take very long)
            knn_train_loss, knn_test_loss = self.test_knn(
                self.full_train_dataset,
                self.full_test_dataset,
                description=f"Task{i}: KNN (Full Test Dataset)"
            )

            knn_train_acc = knn_train_loss.accuracy
            knn_test_acc  = knn_test_loss.accuracy
            logger.info(f"Task{i}: KNN Train Accuracy (Full): {knn_train_acc:.2%}")
            logger.info(f"Task{i}: KNN Test  Accuracy (Full): {knn_test_acc :.2%}")
            self.state.knn_full_losses[i] = knn_test_loss

            # Log the accuracies to wandb.
            self.log({
                f"KNN/train/full": knn_train_acc,
                f"KNN/test/full" : knn_test_acc,
            })

            # Save the state with the new metrics, but no need to save the
            # model weights, as they didn't change.
            self.save_state(save_model_weights=False)
            # NOTE: this has to be after, so we don't reloop through the j's if
            # something in the next few lines fails.
            self.state.j = 0            
            cumul_loss = self.state.cumul_losses[i]
            logger.debug(cumul_loss.dumps(indent="\t"))
            
            cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
            logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
            self.log({
                f"Cumulative": cumul_loss,
            })

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
    
    def load_state(self, state_json_path: Path=None) -> None:
        """ save/restore the state from a previous run. """
        # way to do this.
        logger.info(f"Restoring state from {state_json_path}")        

        if not state_json_path:
            state_json_path = self.checkpoints_dir / "state.json"

        # Load the 'State' object from json
        self.state = self.State.load_json(state_json_path)

        # If any attributes are common to both the Experiment and the State,
        # copy them over to the Experiment.
        for name, (v1, v2) in common_fields(self, self.state):
            logger.info(f"Loaded the {field.name} attribute from the 'State' object.")
            setattr(self, name, v2)

        if self.state.model_weights_path:
            logger.info(f"Restoring model weights from {self.state.model_weights_path}")
            state_dict = torch.load(
                self.state.model_weights_path,
                map_location=self.config.device,
            )
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.info(f"Not restoring model weights (self.state.model_weights_path is None)")

        # TODO: Fix this so the global_step is nicely loaded/restored.
        self.global_step = self.state.global_step or self.state.all_losses.latest_step()
        logger.info(f"Starting at global step = {self.global_step}.")

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

    def load_task_datasets(self, tasks: List[Task]) -> None:
        """Create the train, valid, test, and cumulative valid & test datasets
        for each task.
        """
        # download the dataset.
        train_dataset, test_dataset = super().load_datasets()
        train_full_dataset, valid_full_dataset = self.train_valid_split(train_dataset)
        # safeguard the entire training dataset.
        test_full_dataset = test_dataset
        
        self.full_train_dataset = train_full_dataset
        self.full_valid_dataset = valid_full_dataset
        self.full_test_dataset  = test_full_dataset

        self.train_datasets.clear()
        self.valid_datasets.clear()
        self.test_datasets.clear()

        for i, task in enumerate(tasks):
            train = VisionDatasetSubset(train_full_dataset, classes=task)
            valid = VisionDatasetSubset(valid_full_dataset, classes=task)
            test  = VisionDatasetSubset(test_full_dataset, classes=task)

            self.train_datasets.append(train)
            self.valid_datasets.append(valid)
            self.test_datasets.append(test)

        # Use itertools.accumulate to do the summation of the datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))
        self.test_cumul_dataset = list(accumulate(self.test_datasets))
        
        # if self.config.debug:
        #     for i, (train, valid, test, cumul) in enumerate(zip(self.train_datasets,
        #                                                 self.valid_datasets,
        #                                                 self.test_datasets,
        #                                                 self.valid_cumul_datasets)):
        #         self.save_images(i, train, prefix="train_")
        #         self.save_images(i, valid, prefix="valid_")
        #         self.save_images(i, test, prefix="test_")
        #         self.save_images(i, cumul, prefix="valid_cumul_")
            

    def save_images(self, i: int, dataset: VisionDataset, prefix: str=""):
        n = 64
        samples = dataset.data[:n].view(n, *self.dataset.x_shape).float()
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(samples, self.samples_dir / f"{prefix}task_{i}.png")

    def on_task_switch(self, task: Task, **kwargs) -> None:
        # If we are using a multihead model, we give it the task label (so
        # that it can spawn / reuse the output head for the given task).
        i = self.state.i
        ewc_task = self.model.tasks.get(Tasks.EWC)

        if self.multihead and ewc_task and ewc_task.enabled:
            prev_task = None if i == 0 else self.tasks[i-1]
            classifier_head = None if i == 0 else self.model.get_output_head(prev_task)
            train_loader = self.get_dataloader(self.train_datasets[i])

            if i != 0 and ewc_task.current_task_loader is None:
                previous_task_loader = self.get_dataloader(self.train_datasets[i-1])
                ewc_task.current_task_loader = previous_task_loader

            kwargs.setdefault("prev_task", prev_task)
            kwargs.setdefault("classifier_head", classifier_head)
            kwargs.setdefault("train_loader", train_loader)

        if self.multihead:
            self.model.on_task_switch(task, **kwargs)

    @property
    def started(self) -> bool:
        checkpoint_exists = (self.checkpoints_dir / "state.json").exists()
        return super().started and checkpoint_exists
    
    def log(self, message: Union[str, Dict, LossInfo], **kwargs):  # type: ignore
        if isinstance(message, dict):
            message.setdefault("task/currently_learned_task", self.state.i)
        assert isinstance(message, dict), f"Testing things out, but for now always pass dictionaries to self.log (at least in TaskIncremental)"
        for k, v in message.items():
            if isinstance(v, (LossInfo, Metrics)):
                message[k] = v.to_log_dict()
        
        # Flatten the log dictionary
        from utils.utils import flatten_dict
        flattened = flatten_dict(message)
        
        # TODO: Remove redondant/useless keys
        super().log(flattened, **kwargs)

def get_supervised_accuracy(cumul_loss: LossInfo) -> float:
    # TODO: this is ugly. There is probably a cleaner way, but I can't think of it right now. 
    try:
        return cumul_loss.losses["Test"].losses["supervised"].metrics["supervised"].accuracy
    except KeyError as e:
        print(cumul_loss)
        print(cumul_loss.dumps(indent="\t", sort_keys=False))
        exit()
        raise e


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TaskIncremental, dest="experiment")
    
    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    
    from main import launch
    launch(experiment)
