import os
import pprint
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
from models.classifier import Classifier
from tasks import AuxiliaryTask
from tasks.reconstruction.vae import VAEReconstructionTask

from .experiment import Experiment


@dataclass
class IID(Experiment):
    """ Simple IID setting. """
    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_task: Optional[VAEReconstructionTask] = None
        
        # If the model is a SelfSupervised classifier, it will have a `tasks` attribute.
        # find the reconstruction task, if there is one.
        if "reconstruction" in self.model.tasks:
            self.reconstruction_task = self.model.tasks["reconstruction"]
            self.latents_batch = torch.randn(64, self.hparams.hidden_size)
    
    def run(self):
        self.load()
        train_losses, valid_losses = self.train_until_convergence(self.dataset.train, self.dataset.valid, self.hparams.epochs)
        # make the training plots
        plots_dict = self.make_plots(train_losses, valid_losses)

        # Get the most recent validation metrics. 
        last_step = max(valid_losses.keys())
        last_val_loss = valid_losses[last_step]
        class_accuracy = last_val_loss.metrics.class_accuracy
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

        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=30)
        plots_dict["losses"] = fig
        fig.savefig(self.plots_dir / "losses.jpg")

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.plot(list(train_losses.keys()), [l.metrics.accuracy for l in train_losses.values()], label="train")
        ax.plot(list(valid_losses.keys()), [l.metrics.accuracy for l in valid_losses.values()], label="valid")
        ax.legend(loc='lower right')
        
        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=30)

        plots_dict["accuracy"] = fig
        fig.savefig(self.plots_dir / "accuracy.jpg")
        return plots_dict


    def test_iter(self, dataloader: DataLoader):
        yield from super().test_iter(dataloader)
        if self.reconstruction_task:
            self.generate_samples()
    
    def train_iter(self, dataloader: DataLoader):
        for loss_info in super().train_iter(dataloader):
            yield loss_info
        
        if self.reconstruction_task:
            # use the last batch of x's.
            x_batch = loss_info.tensors.get("x")
            if x_batch is not None:
                self.reconstruct_samples(x_batch)
        
        
    def reconstruct_samples(self, data: Tensor):
        with torch.no_grad():
            n = min(data.size(0), 8)
            
            originals = data[:n]
            reconstructed = self.reconstruction_task.reconstruct(originals)
            comparison = torch.cat([originals, reconstructed])

            reconstruction_images_dir = self.config.log_dir / "reconstruction"
            reconstruction_images_dir.mkdir(exist_ok=True)
            file_name = reconstruction_images_dir / f"reconstruction_step_{self.global_step}.png"
            
            save_image(comparison.cpu(), file_name, nrow=n)

    def generate_samples(self):
        with torch.no_grad():
            n = 64
            fake_samples = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
            fake_samples = fake_samples.cpu().view(n, *self.dataset.x_shape)

            generation_images_dir = self.config.log_dir / "generated_samples"
            generation_images_dir.mkdir(exist_ok=True)
            file_name = generation_images_dir / f"generated_step_{self.global_step}.png"
            save_image(fake_samples, file_name)

