import os
import pprint
import sys
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type

import matplotlib.pyplot as plt
import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import ArgumentParser, choice, field, subparsers
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common.losses import LossInfo
from common.metrics import get_metrics
from config import Config
from datasets.dataset import Dataset
from datasets.mnist import Mnist
from experiment import Experiment
from models.classifier import Classifier
from tasks import AuxiliaryTask, Tasks


@dataclass
class IID(Experiment):
    """ Simple IID setting. """
    def __post_init__(self):
        super().__post_init__()
    
    def run(self):
        self.load_datasets()
        self.model = self.init_model()
        train_losses, valid_losses = self.train_until_convergence(self.dataset.train, self.dataset.valid, self.hparams.epochs)
        # make the training plots
        plots_dict = self.make_plots(train_losses, valid_losses)

        for figure_name, fig in plots_dict.items():    
            if self.config.debug:
                fig.show()
                fig.waitforbuttonpress(timeout=30)
            fig.savefig(self.plots_dir / Path(figure_name).with_suffix(".jpg"))

        # Get the most recent validation metrics. 
        last_step = max(valid_losses.keys())
        last_val_loss = valid_losses[last_step]
        class_accuracy = last_val_loss.metrics[Tasks.SUPERVISED].class_accuracy
        valid_class_accuracy_mean = class_accuracy.mean()
        valid_class_accuracy_std = class_accuracy.std()
        self.log("Validation Average Class Accuracy: ", valid_class_accuracy_mean, once=True, always_print=True)
        self.log("Validation Class Accuracy STD:", valid_class_accuracy_std, once=True, always_print=True)
        self.log(plots_dict, once=True)

        return train_losses, valid_losses

    def make_plots(self, train_losses: Dict[int, LossInfo], valid_losses: Dict[int, LossInfo]) -> Dict[str, plt.Figure]:
        plots_dict: Dict[str, plt.Figure] = {}

        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()
        ax.set_title("Total Loss")
        ax.set_xlabel("# of Samples seen")
        ax.set_ylabel("Loss")
        ax.plot(list(train_losses.keys()), [l.total_loss for l in train_losses.values()], label="train")
        ax.plot(list(valid_losses.keys()), [l.total_loss for l in valid_losses.values()], label="valid")
        ax.legend(loc="upper right")
        plots_dict["losses"] = fig

        # TODO: add the loss plots for all the auxiliary tasks here?

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        x = list(train_losses.keys())
        y_train = [l.metrics[Tasks.SUPERVISED].accuracy for l in train_losses.values()]
        y_valid = [l.metrics[Tasks.SUPERVISED].accuracy for l in valid_losses.values()]
        ax.plot(x, y_train, label="train")
        ax.plot(x, y_valid, label="valid")
        ax.legend(loc='lower right')
        plots_dict["accuracy"] = fig
        return plots_dict


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    
    args = parser.parse_args()
    experiment: IID = args.experiment
    
    from main import launch
    launch(experiment)
