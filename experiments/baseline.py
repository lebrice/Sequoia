import os
import pprint
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, InitVar
from typing import Any, Dict, Iterable, List, Tuple, Type, ClassVar

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

from config import Config
from datasets.bases import Dataset
from datasets.mnist import Mnist
from experiments.experiment import Experiment
from models.classifier import Classifier, MnistClassifier
from utils.logging import loss_str
from common.losses import LossInfo


@dataclass
class Baseline(Experiment):
    dataset: Dataset = choice({
        "mnist": Mnist(),
    }, default="mnist")

    hparams: Classifier.HParams = Classifier.HParams()
    config: Config = Config()

    def load(self, config):
        if isinstance(self.dataset, Mnist):
            self.model = MnistClassifier(
                hparams=self.hparams,
                config=self.config,
            )
        else:
            raise NotImplementedError("TODO: add other datasets.")
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders

    def run(self):
        self.load(self.config)

        train_epoch_loss: List[LossInfo] = []
        valid_epoch_loss: List[LossInfo] = []

        for epoch in range(self.hparams.epochs):
            for train_loss in self.train_iter(epoch, self.train_loader):
                pass
            train_epoch_loss.append(train_loss)
            
            for valid_loss in self.test_iter(epoch, self.valid_loader):
                pass
            valid_epoch_loss.append(valid_loss)

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
