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
        dataloaders = self.dataset.get_dataloaders(self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders

    def run(self):
        for epoch in range(self.hparams.epochs):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            with torch.no_grad():
                sample = model.generate(torch.randn(64, hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                save_image(sample, 'results/sample_' + str(epoch) + '.png')
    
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

    def test_epoch(self, epoch: int):
        model = self.model
        dataloader = self.valid_loader
        model.eval()
        test_loss = 0.

        overall_loss_info = LossInfo()

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(model.device)
                batch_loss_info = model.get_loss(data, target)

                if i == 0:
                    n = min(data.size(0), 8)
                    fake = model.reconstruct(data)
                    # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
                    comparison = torch.cat([data[:n], fake[:n]])
                    save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)

        test_loss /= len(dataloader.dataset)
        print(f"====> Test set loss: {test_loss:.4f}")
        print(*[f"{loss_name}: {loss_str(loss_tensor)}" for loss_name, loss_tensor in losses.items()], sep=" ")
