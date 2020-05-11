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
from torch import Tensor

from common.losses import LossInfo
from datasets.subset import VisionDataset
from task_incremental import TaskIncremental
from utils.json_utils import try_load
from utils.plotting import PlotSectionLabel

from common.losses import TrainValidLosses
TrainAndValidLosses = TrainValidLosses


@dataclass
class ActiveRemembering(TaskIncremental):
    """ Active-remembering experiment. Can we "remember" without labels?
    
    TODO: Add your arguments as attributes here, if any.
    """
    unsupervised_epochs_per_task: int = 0

    # The maximum number of epochs to train on when remembering without labels.
    remembering_max_epochs: int = 1

    def run(self):
        # Load the datasets and return the set of classes within each task.
        tasks: List[List[int]] = self.load()

        self.model = self.init_model()
        
        if not any(task.enabled for task in self.model.tasks.values()):
            self.log("At least one Auxiliary Task should be activated in order "
                     "to do active remembering!\n"
                     "(Set one with '--<task>.coefficient <value>').")
            exit()

        # All classes in the order in which they appear
        label_order: List[int] = sum(tasks, [])
        print("Class Ordering:", label_order)
        
        train_valid_losses = (
            TrainValidLosses.try_load_json(self.results_dir / "losses.json") or
            TrainValidLosses()
        )

        self.global_step = train_valid_losses.latest_step()
        if self.global_step != 0:
            self.plot_sections = try_load(self.results_dir / "plot_labels.pt", [])
            # TODO: reset the state of the experiment.
            print(f"Experiment is already at step {self.global_step}")
            # Right now I just skip the training and just go straight to making the plot with the existing data:
            self.tasks = []

        for task_index, task in enumerate(self.tasks):
            self.logger.info(f"Starting task {task_index} with classes {task}")
            
            train_i: VisionDatasetSubset = self.train_datasets[task_index]
            valid_i: VisionDatasetSubset = self.valid_datasets[task_index]
            valid_0_to_i: VisionDatasetSubset = self.valid_cumul_datasets[task_index]

            # If we are using a multihead model, we give it the task label (so
            # that it can spawn / reuse the output head for the given task).
            if self.multihead:
                self.model.current_task_id = task_index

            with self.plot_region_name(f"Learn Task {task_index}"):
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

                train_0: VisionDatasetSubset = self.train_datasets[0]
                valid_0: VisionDatasetSubset = self.valid_datasets[0]

                with train_0.without_labels(), self.plot_region_name("Remember Task 0"):
                    # Here by using train_until_convergence we also periodically
                    # evaluate the validation loss on batches from the (labeled)
                    # validation set.
                    train_valid_losses += self.train_until_convergence(
                        train_0,
                        valid_0,
                        max_epochs=self.remembering_max_epochs,
                        description=f"Task 0 Remembering (Unsupervised)"
                    )

        # TODO: Save the results to a json file.
        train_valid_losses.save_json(self.results_dir / "losses.json")
        torch.save(self.plot_sections, str(self.results_dir / "plot_labels.pt"))

        fig = make_plot(train_valid_losses, self.plot_sections)
        
        from utils.plotting import maximize_figure
        maximize_figure()
        
        fig.savefig(self.plots_dir / "remembering_plot.png")
        
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
            if loss_name in loss_info.losses:
                task_loss = loss_info.losses[loss_name]
                y = task_loss.total_loss.item()
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
