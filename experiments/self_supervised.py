import os
import pprint
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Type

import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import ArgumentParser, field, subparsers, choice, list_field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common.losses import LossInfo
from config import Config
from datasets.mnist import Mnist
from experiments.experiment import Experiment
from experiments.iid import IID
from models.ss_classifier import MnistClassifier, SelfSupervisedClassifier

from tasks.torchvision.adjust_brightness import AdjustBrightnessTask
from utils.logging import loss_str
from tasks import AuxiliaryTask, VAEReconstructionTask


@dataclass
class SelfSupervised(IID):
    """ Simply adds auxiliary tasks to the IID experiment. """
    hparams: SelfSupervisedClassifier.HParams = SelfSupervisedClassifier.HParams(detach_classifier=False)

    model: SelfSupervisedClassifier = field(default=None, init=False)
    reconstruction_task: Optional[VAEReconstructionTask] = field(default=None, init=False)

    def __post_init__(self):
        AuxiliaryTask.input_shape   = self.dataset.x_shape
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size

        if isinstance(self.dataset, Mnist):
            from models.ss_classifier import MnistClassifier
            self.model = MnistClassifier(
                hparams=self.hparams,
                config=self.config,
                tasks=self.hparams.get_tasks(),
            )
        else:
            raise NotImplementedError("TODO: add other datasets.")
        
        # find the reconstruction task, if there is one.
        for aux_task in self.model.tasks:
            if isinstance(aux_task, VAEReconstructionTask):
                self.reconstruction_task = aux_task
                break

        dataloaders = self.dataset.get_dataloaders(self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders

    def run(self):
        train_epoch_loss: List[LossInfo] = []
        valid_epoch_loss: List[LossInfo] = []

        for epoch in range(self.hparams.epochs):
            for train_loss in self.train_iter(epoch, self.train_loader):
                pass
            train_epoch_loss.append(train_loss)
            
            for valid_loss in self.test_iter(epoch, self.valid_loader):
                pass
            valid_epoch_loss.append(valid_loss)

            if self.reconstruction_task:
                with torch.no_grad():
                    sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                    sample = sample.cpu().view(64, 1, 28, 28)
                    save_image(sample, os.path.join(self.config.log_dir, f"sample_{epoch}.png"))

            if self.config.wandb:
                # TODO: do some nice logging to wandb?:
                wandb.log(TODO)
        
        import matplotlib.pyplot as plt
        fig: plt.Figure = plt.figure()
        plt.plot([loss.total_loss for loss in train_epoch_loss], label="train_loss")
        plt.plot([loss.total_loss for loss in valid_epoch_loss], label="valid_loss")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_loss.jpg"))


        fig: plt.Figure = plt.figure()
        plt.plot([loss.metrics.accuracy for loss in train_epoch_loss], label="train_accuracy")
        plt.plot([loss.metrics.accuracy for loss in valid_epoch_loss], label="valid_accuracy")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_accuracy.jpg"))

    
    def log_info(self, batch_loss_info: LossInfo, overall_loss_info: LossInfo) -> Dict:
        message: Dict[str, Any] = super().log_info(batch_loss_info, overall_loss_info)
        for loss_name, loss_tensor in batch_loss_info.losses.items():
            if loss_name.endswith("_scaled"):
                continue
            scaled_loss_tensor = batch_loss_info.losses.get(f"{loss_name}_scaled")
            if scaled_loss_tensor is not None:
                message[loss_name] = f"{loss_str(scaled_loss_tensor)} ({loss_str(loss_tensor)})"
            else:
                message[loss_name] = loss_str(loss_tensor)
        return message

# if i == 0:
            #     n = min(data.size(0), 8)
            #     fake = self.reconstruction_task.reconstruct(data)
            #     # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
            #     comparison = torch.cat([data[:n], fake[:n]])
            #     save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)
