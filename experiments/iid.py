from utils.logging_utils import get_logger
import os
import pprint
import sys
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm
import wandb
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from simple_parsing import ArgumentParser, mutable_field
from common.losses import LossInfo, TrainValidLosses
from common.metrics import get_metrics
from datasets.dataset import Dataset
from datasets.mnist import Mnist
from models.classifier import Classifier
from simple_parsing import ArgumentParser, choice, field, subparsers
from tasks import AuxiliaryTask, Tasks

from .experiment import Experiment

logger = get_logger(__file__)


@dataclass
class IID(Experiment):
    """ Simple IID setting. """
    @dataclass
    class Config(Experiment.Config):
        """ Config for the IID experiment. """
        # Maximum number of epochs to train for. 
        max_epochs: int = 10
    
    config: Config = mutable_field(Config)

    def run(self) -> Tuple[TrainValidLosses, LossInfo]:
        """ Simple IID Training on train/valid datasets, then evaluate on test dataset. """
        self.setup()
        train, valid, test = self.load_datasets()
        assert valid is not None
        # Get the dataloaders
        train_loader = self.get_dataloader(train)
        valid_loader = self.get_dataloader(valid)
        test_loader = self.get_dataloader(test)

        # Train until convergence on validation set (or for a number of epochs) 
        all_losses = self.train(
            train_loader,
            valid_loader,
            epochs=self.config.max_epochs,
            temp_save_dir=self.checkpoints_dir,
        )
        # Save to results dir.
        self.results_dir.mkdir(exist_ok=True)
        self.save_state(self.results_dir)

        test_loss = self.test(test_loader)
        self.log({"Test": test_loss}, once=True)
        
        if self.config.use_wandb:
            wandb.run.summary["Test loss"] = test_loss.losses[Tasks.SUPERVISED].total_loss
            wandb.run.summary["Test Accuracy"] = test_loss.losses[Tasks.SUPERVISED].accuracy
        # make training/validation plots. Not really needed when using wandb.
        plots_dict = self.make_plots(all_losses)

        for figure_name, fig in plots_dict.items():    
            if self.config.debug:
                fig.show()
                fig.waitforbuttonpress(timeout=30)
            fig.savefig(self.plots_dir / Path(figure_name).with_suffix(".jpg"))

        # Get the most recent validation metrics. 
        last_step = max(all_losses.valid_losses.keys())
        last_val_loss = all_losses.valid_losses[last_step]
        class_accuracy = last_val_loss.losses[Tasks.SUPERVISED].metric.class_accuracy
        valid_class_accuracy_mean = class_accuracy.mean()
        valid_class_accuracy_std = class_accuracy.std()
        logger.info(f"Validation Average Class Accuracy: {valid_class_accuracy_mean:.2%}")
        logger.info(f"Validation Class Accuracy STD: {valid_class_accuracy_std}")
        self.log(plots_dict, once=True)

        return all_losses, test_loss

    def make_plots(self, all_losses: TrainValidLosses) -> Dict[str, plt.Figure]:
        train_losses: Dict[int, LossInfo] = all_losses.train_losses
        valid_losses: Dict[int, LossInfo] = all_losses.valid_losses
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
        y_train = [l.losses[Tasks.SUPERVISED].accuracy for l in train_losses.values()]
        y_valid = [l.losses[Tasks.SUPERVISED].accuracy for l in valid_losses.values()]
        ax.plot(x, y_train, label="train")
        ax.plot(x, y_valid, label="valid")
        ax.legend(loc='lower right')
        plots_dict["accuracy"] = fig
        return plots_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(IID, dest="experiment")
    
    args = parser.parse_args()
    experiment: IID = args.experiment
    experiment.launch()
