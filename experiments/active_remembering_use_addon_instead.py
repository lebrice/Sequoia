import functools
import itertools
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import Tensor

from common.losses import LossInfo, TrainValidLosses
from common.task import Task
from datasets.data_utils import unlabeled
from datasets.subset import Dataset
from simple_parsing import mutable_field
from utils.logging_utils import get_logger
from utils.plotting import PlotSectionLabel
from datasets.data_utils import unlabeled

from .task_incremental import TaskIncremental
from datasets import Datasets
logger = get_logger(__file__)


@dataclass
class ActiveRemembering(TaskIncremental):
    """ Active-remembering experiment. Can we "remember" without labels?
    
    TODO: Add your arguments as attributes here, if any.
    """
    @dataclass
    class Config(TaskIncremental.Config):
        """Config for the active remembering experiment. """
        # The maximum number of epochs to train on when remembering without labels.
        remembering_max_epochs: int = 0

    config: Config = mutable_field(Config)

    def run(self):
        """TODO: Evaluate remembering on the test set? validation? Report online acc? or single value? Test time training or no?

        Returns:
            [type]: [description]
        """
        self.setup()
        # Load the datasets and return the set of classes within each task.
        tasks: List[Task] = self.state.tasks
        # All classes in the order in which they appear
        print("Class Ordering:", sum((t.classes for t in tasks), []))

        self.model = self.init_model()
        if not any(task.enabled for task in self.model.tasks.values()):
            raise RuntimeError(
                "At least one Auxiliary Task should be activated in order to "
                "do active remembering!\n"
                "(Set one with '--<task>.coefficient <value>')."
            )

        all_losses = self.state.all_losses

        if self.state.global_step != 0:
            # TODO: reset the state of the experiment.
            print(f"Experiment is already at step {self.global_step}")
            # Right now I just skip the training and just go straight to making the plot with the existing data:
            self.tasks = []

        for i, task in enumerate(self.tasks):
            self.state.i = i

            task_index = i
            logger.info(f"Starting task {task_index} with classes {task}")
            
            train_i: VisionDatasetSubset = self.train_datasets[task_index]
            valid_i: VisionDatasetSubset = self.valid_datasets[task_index]
            train_i_loader = self.get_dataloader(train_i)
            valid_i_loader = self.get_dataloader(valid_i)

            valid_0_to_i: VisionDatasetSubset = self.valid_cumul_datasets[task_index]

            self.on_task_switch(task)

            with self.plot_region_name(f"Learn Task {task_index}"):
                # Un/self-supervised training on task i.
                all_losses += self.train(
                    unlabeled(train_i_loader),
                    unlabeled(valid_i_loader),
                    epochs=self.config.unsupervised_epochs_per_task,
                    description=f"Task {i} (Unsupervised)",
                    temp_save_dir=self.checkpoints_dir / f"task_{i}_unsupervised",
                )

                # Train (supervised) on task i.
                all_losses += self.train(
                    train_i_loader,
                    valid_i_loader,
                    epochs=self.config.supervised_epochs_per_task,
                    description=f"Task {i} (Supervised)",
                    temp_save_dir=self.checkpoints_dir / f"task_{i}_supervised",
                )
                
            # Actively "remember" task 0 by training with self-supervised on it.
            if task_index >= 1:
                # Use the output head for task 0 if we are in multihead setup:
                self.on_task_switch(tasks[0])

                train_0: VisionDatasetSubset = self.train_datasets[0]
                valid_0: VisionDatasetSubset = self.valid_datasets[0]
                train_0_loader = self.get_dataloader(train_0)
                valid_0_loader = self.get_dataloader(valid_0)
                
                with self.plot_region_name("Remember Task 0"):
                    # Here by using train we also periodically
                    # evaluate the validation loss on batches from the (labeled)
                    # validation set.
                    all_losses += self.train(
                        unlabeled(train_0_loader),
                        valid_0_loader,
                        epochs=self.config.remembering_max_epochs,
                        description=f"Task 0 Remembering (Unsupervised)",
                        temp_save_dir=self.checkpoints_dir / f"task_{task_index}_remembering",
                    )
            
            self.save_state()

        self.save_state(self.results_dir)
        fig = make_plot(all_losses, self.state.plot_sections)
        
        from utils.plotting import maximize_figure
        maximize_figure()
        
        fig.savefig(self.plots_dir / "remembering_plot.png")
        if self.config.use_wandb:
            wandb.log({"Rememering": fig})

        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(10)
            
        
def make_plot(train_and_valid_losses: TrainValidLosses,
              plot_sections: List[PlotSectionLabel]=None) -> plt.Figure:
    train_losses: Dict[int, LossInfo] = train_and_valid_losses.train_losses
    valid_losses: Dict[int, LossInfo] = train_and_valid_losses.valid_losses
    
    plot_sections = plot_sections or []

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.subplots()
    ax.set_title("Total Loss")
    ax.set_xlabel("# of Samples seen")
    ax.set_ylabel("Loss")
    
    # Figure out the name of all the losses that were used.
    all_loss_names: Set[str] = train_and_valid_losses.all_loss_names()
    print("All loss names:", all_loss_names)

    def get_x_y(loss_dict: Dict[int, LossInfo],
                loss_name: str) -> Tuple[List[int], List[Optional[float]]]:
        xs: List[int] = []
        ys: List[Optional[float]] = []
        for step, loss_info in loss_dict.items():
            x = step
            y = None
            from common.losses import get_supervised_accuracy
            y = get_supervised_accuracy(loss_info.losses[loss_name])
            xs.append(x)
            ys.append(y)
        return xs, ys

    plot_data: Dict[str, Tuple[List[int], List[Optional[float]]]] = {}
    for loss_name in all_loss_names:
        x_train, y_train = get_x_y(train_losses, loss_name)
        x_valid, y_valid = get_x_y(valid_losses, loss_name)
        label = f"Train {loss_name}"
        plot_data[label] = (x_train, y_train)
        label=f"Valid {loss_name}"
        plot_data[label] = (x_valid, y_valid)
    
    for section_label in plot_sections:
        for loss_name, (x, y) in plot_data.items():
            # insert a (x, None) pair before the vertical line, so it
            # doesn't jump up (so that no plot lines cross the vertical
            # line)
            x.append(section_label.start_step-1)
            y.append(None)

        # Add vertical lines at the start_step and end_step of the label
        # along with the description in between.
        section_label.annotate(ax)

    for label, (x, y) in plot_data.items():
        sort_index = np.argsort(x)
        xs = np.asarray(x)
        ys = np.asarray(y)
        # sort the pairs by x to place the potential (x, None) pairs
        ax.plot(xs[sort_index], ys[sort_index], label=label)

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
