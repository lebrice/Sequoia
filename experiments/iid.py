import os
import pprint
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type

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
from config import Config
from datasets.bases import Dataset
from datasets.mnist import Mnist
from experiments.experiment import Experiment
from models.classifier import Classifier
from tasks import AuxiliaryTask
from utils.logging import loss_str

from models.ss_classifier import SelfSupervisedClassifier
from tasks.reconstruction import VAEReconstructionTask


@dataclass
class IID(Experiment):
    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_task: Optional[VAEReconstructionTask] = None
        
        # If the model is a SelfSupervised classifier, it will have a `tasks` attribute.
        # find the reconstruction task, if there is one.
        if isinstance(self.model, SelfSupervisedClassifier):
            for aux_task in self.model.tasks:
                if isinstance(aux_task, VAEReconstructionTask):
                    self.reconstruction_task = aux_task
                    break
    
    def make_plots_for_epoch(self, epoch: int, train_losses: List[LossInfo], valid_losses: List[LossInfo]):
        import matplotlib.pyplot as plt
        fig: plt.Figure = plt.figure()
        x_s: List[int] = []
        for loss_info in train_losses:
            x_s = [loss.metrics.n_samples for loss in train_losses]
        

        plt.plot([loss.total_loss for loss in train_losses], label="train_loss")
        plt.plot([loss.total_loss for loss in valid_losses], label="valid_loss")
        plt.legend(loc='lower right')
        fig.savefig(self.config.log_dir + "/train/epoch_{epoch}_loss.jpg")

        fig = plt.figure()
        plt.plot([loss.metrics.accuracy for loss in train_losses], label="train_accuracy")
        plt.plot([loss.metrics.accuracy for loss in valid_losses], label="valid_accuracy")
        plt.legend(loc='lower right')
        fig.savefig(self.config.log_dir + "/train/epoch_{epoch}_acc.jpg")
        

    def make_plots(self, train_epoch_loss: List[LossInfo], valid_epoch_loss: List[LossInfo]):
        
        # TODO: (Currently under construction, will create plots for each epoch)
        return
        
        import matplotlib.pyplot as plt
        fig: plt.Figure = plt.figure()
        ax1: plt.Axes = fig.add_subplot(nrows=1, ncols=2, index=1)
        ax1.set_xlabel("# of Samples")
        ax1.plot([loss.total_loss for loss in train_epoch_loss], label="train_loss")
        ax1.plot([loss.total_loss for loss in valid_epoch_loss], label="valid_loss")
        ax1.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_loss.jpg"))

        fig = plt.figure()
        plt.plot([loss.metrics.accuracy for loss in train_epoch_loss], label="train_accuracy")
        plt.plot([loss.metrics.accuracy for loss in valid_epoch_loss], label="valid_accuracy")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_accuracy.jpg"))

    def test_iter(self, epoch: int, dataloader: DataLoader):
        yield from super().test_iter(epoch, dataloader)
        if self.reconstruction_task:
            with torch.no_grad():
                sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                save_image(sample, os.path.join(self.config.log_dir, f"sample_{epoch}.png"))
    
    def train_iter(self, epoch: int, dataloader: DataLoader):
        yield from super().train_iter(epoch, dataloader)
        if self.reconstruction_task:
            with torch.no_grad():
                sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                os.makedirs(os.path.join(self.config.log_dir, "generated"))
                save_image(sample, os.path.join(self.config.log_dir, "generated", f"{epoch}.png"))

                # os.makedirs(os.path.join(self.config.log_dir, "reconstruction"))
                # n = min(data.size(0), 8)
                # fake = self.reconstruction_task.reconstruct()
                # comparison = torch.cat([data[:n], fake[:n]])
                # save_image(comparison.cpu(), os.path.join(self.config.log_dir, "reconstruction", f"{epoch}.png"), nrow=n)

    def log_info(self, batch_loss_info: LossInfo, overall_loss_info: LossInfo) -> Dict:
        message = super().log_info(batch_loss_info, overall_loss_info)
        # add the logs for all the scaled losses:
        for loss_name, loss_tensor in batch_loss_info.losses.items():
            if loss_name.endswith("_scaled"):
                continue
            scaled_loss_tensor = batch_loss_info.losses.get(f"{loss_name}_scaled")
            if scaled_loss_tensor is not None:
                message[loss_name] = f"{loss_str(scaled_loss_tensor)} ({loss_str(loss_tensor)})"
            else:
                message[loss_name] = loss_str(loss_tensor)
        return message