from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple, Optional, Set

import matplotlib.pyplot as plt

import functools
import itertools
from common.losses import LossInfo
from datasets.subset import VisionDataset
from task_incremental import TaskIncremental
from contextlib import contextmanager
from torch import Tensor
from pathlib import Path

@dataclass
class TrainAndValidLosses:
    """ Helper class to store the train and valid losses during training. """
    train_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)
    valid_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)

    def __iadd__(self, other: Union["TrainAndValidLosses", Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]]) -> "TrainAndValidLosses":
        if isinstance(other, TrainAndValidLosses):
            self.train_losses.update(other.train_losses)
            self.valid_losses.update(other.valid_losses)
            return self
        elif isinstance(other, tuple):
            self.train_losses.update(other[0])
            self.valid_losses.update(other[1])
            return self
        else:
            return NotImplemented

    def all_loss_names(self) -> Set[str]:
        all_loss_names: Set[str] = set()
        for loss_info in itertools.chain(self.train_losses.values(), 
                                         self.valid_losses.values()):
            all_loss_names.update(loss_info.losses)
        return all_loss_names

    def save_json(self, path: Path) -> None:
        """ TODO: save to a json file. """
        pass


@dataclass
class ActiveRemembering(TaskIncremental):
    """ Active-remembering experiment. Can we "remember" without labels?
    
    TODO: Add your arguments as attributes here, if any.
    """
    # The maximum number of epochs to train on when remembering without labels.
    remembering_max_epochs: int = 1

    def run(self):
        # Load the datasets and return the set of classes within each task.
        tasks: List[List[int]] = self.load()

        self.init_model()
        
        if not any(task.enabled for task in self.model.tasks.values()):
            self.log("At least one Auxiliary Task should be activated in order "
                     "to do active remembering!\n"
                     "(Set one with '--<task>.coefficient <value>').")
            exit()

        # All classes in the order in which they appear
        label_order: List[int] = sum(tasks, [])
        print("Class Ordering:", label_order)
        
        datasets = zip(
            self.train_datasets,
            self.valid_datasets,
            self.valid_cumul_datasets
        )
        
        train_losses: List[LossInfo] = []
        valid_losses: List[LossInfo] = []

        train_i: VisionDatasetSubset
        valid_i: VisionDatasetSubset
        valid_0_to_i: VisionDatasetSubset
        
        train_0: VisionDatasetSubset = self.train_datasets[0]
        valid_0: VisionDatasetSubset = self.valid_datasets[0]

        train_valid_losses = TrainAndValidLosses()

        for task_index, (train_i, valid_i, valid_0_to_i) in enumerate(datasets):
            print(f"Starting task {task_index} with classes {tasks[task_index]}")
            
            # If we are using a multihead model, we give it the task label (so
            # that it can spawn / reuse the output head for the given task).
            if self.multihead:
                self.model.current_task_id = task_index
            
            # Temporarily remove the labels.
            with train_i.without_labels(), valid_i.without_labels():
                # Un/self-supervised training on task i.
                train_valid_losses += self.train_until_convergence(
                    train_i,
                    valid_i,
                    max_epochs=self.unsupervised_epochs_per_task,
                    description=f"Task {task_index} (Unsupervised)",
                )

            # Train (supervised) on task i.
            train_valid_losses += self.train_until_convergence(
                train_i,
                valid_i,
                max_epochs=self.supervised_epochs_per_task,
                description=f"Task {task_index} (Supervised)",
            )
            
            # Actively "remember" task 0 by training with self-supervised on it.
            if task_index >= 1:
                # Use the output head for task 0 if we are in multihead setup:
                if self.multihead:
                    self.model.current_task_id = 0
                with train_0.without_labels():
                    # Here by using train_until_convergence we also periodically
                    # evaluate the validation loss on batches from the (labeled)
                    # validation set.
                    train_valid_losses += self.train_until_convergence(
                        train_0,
                        valid_0,
                        max_epochs=self.remembering_max_epochs,
                        description=f"Task 0 Remembering (Unsupervised)"
                    )

        fig = self.make_plot(train_valid_losses)
        fig.savefig(self.plots_dir / "remembering_plot.png")
        if self.config.debug:
            fig.show()
        
    def make_plot(self, train_and_valid_losses: TrainAndValidLosses) -> plt.Figure:
        train_losses: Dict[int, LossInfo] = train_and_valid_losses.train_losses
        valid_losses: Dict[int, LossInfo] = train_and_valid_losses.valid_losses
        
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()
        ax.set_title("Total Loss")
        ax.set_xlabel("# of Samples seen")
        ax.set_ylabel("Loss")
        
        # Figure out the name of all the losses that were used.
        all_loss_names: Set[str] = train_and_valid_losses.all_loss_names()
        print("All loss names:", all_loss_names)

        def get_x_y(loss_dict: Dict[int, LossInfo],
                    loss_name: str) -> Tuple[List[int], List[float]]:
            xs: List[int] = []
            ys: List[float] = []
            for step, loss_info in loss_dict.items():
                if loss_name in loss_info.losses:
                    x = step
                    y = loss_info.losses[loss_name].total_loss
                    xs.append(x)
                    ys.append(y.item())
            return xs, ys

        for loss_name in all_loss_names:
            x_train, y_train = get_x_y(train_losses, loss_name)
            x_valid, y_valid = get_x_y(valid_losses, loss_name)
            ax.plot(x_train, y_train, label=f"Train {loss_name}")
            ax.plot(x_valid, y_valid, label=f"Valid {loss_name}")       
        
        ax.legend(loc="upper right")
        
        return fig

        # TODO: maybe show evolution of accuracy in another subfigure?
        # fig, ax = plt.subplots()
        # ax.set_xlabel("Epoch")
        # ax.set_ylim(0.0, 1.0)
        # ax.set_ylabel("Accuracy")
        # ax.set_title("Training and Validation Accuracy")
        # x = list(train_losses.keys())
        # from tasks.tasks import Tasks
        # y_train = [l.metrics[Tasks.SUPERVISED].accuracy for l in train_losses.values()]
        # y_valid = [l.metrics[Tasks.SUPERVISED].accuracy for l in valid_losses.values()]
        # ax.plot(x, y_train, label="train")
        # ax.plot(x, y_valid, label="valid")
        # ax.legend(loc='lower right')


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(ActiveRemembering, dest="experiment")
    
    args = parser.parse_args()
    experiment: ActiveRemembering = args.experiment
    
    from main import launch
    launch(experiment)
