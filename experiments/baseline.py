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


@dataclass
class Baseline(Experiment):
    def make_plots(self, train_epoch_loss: List[LossInfo], valid_epoch_loss: List[LossInfo]):
        import matplotlib.pyplot as plt
        fig: plt.Figure = plt.figure()
        plt.plot([loss.total_loss for loss in train_epoch_loss], label="train_loss")
        plt.plot([loss.total_loss for loss in valid_epoch_loss], label="valid_loss")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_loss.jpg"))


        fig = plt.figure()
        plt.plot([loss.metrics.accuracy for loss in train_epoch_loss], label="train_accuracy")
        plt.plot([loss.metrics.accuracy for loss in valid_epoch_loss], label="valid_accuracy")
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(self.config.log_dir, "epoch_accuracy.jpg"))