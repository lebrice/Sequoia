import pprint
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import ArgumentParser, field, subparsers
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.mnist import Mnist
from models.bases import Config, Model
from models.semi_supervised.classifier import HParams, SelfSupervisedClassifier
from tasks import (AuxiliaryTask, IrmTask, JigsawPuzzleTask, ManifoldMixupTask,
                   MixupTask, PatchLocationTask, RotationTask, TaskType,
                   VAEReconstructionTask)
from tasks.torchvision.adjust_brightness import AdjustBrightnessTask
from utils.logging import loss_str

from .experiment import Experiment


@dataclass
class HParamsWithAuxiliaryTasks(HParams):
    """ Set of Options / Command-line Parameters for the auxiliary tasks. """
    reconstruction: VAEReconstructionTask.Options = VAEReconstructionTask.Options(coefficient=0.001)
    mixup:          MixupTask.Options = MixupTask.Options(coefficient=0.001)
    manifold_mixup: ManifoldMixupTask.Options = ManifoldMixupTask.Options(coefficient=0.1)
    rotation:       RotationTask.Options = RotationTask.Options(coefficient=0.1)
    jigsaw:         JigsawPuzzleTask.Options = JigsawPuzzleTask.Options(coefficient=0)
    irm:            IrmTask.Options = IrmTask.Options(coefficient=1)
    adjust_brightness: AdjustBrightnessTask.Options = AdjustBrightnessTask.Options(coefficient=1.0)

    def get_tasks(self):
        tasks = []
        tasks.append(RotationTask(options=self.rotation))
        tasks.append(JigsawPuzzleTask(options=self.jigsaw))
        tasks.append(ManifoldMixupTask(options=self.manifold_mixup))
        tasks.append(MixupTask(options=self.mixup))
        tasks.append(IrmTask(options=self.irm))
        tasks.append(AdjustBrightnessTask(options=self.adjust_brightness))
        return tasks


@dataclass
class MnistSSL(Experiment):
    dataset: Mnist = Mnist(iid=True)
    hparams: HParamsWithAuxiliaryTasks = HParamsWithAuxiliaryTasks()
    config: Config = Config()

    model: SelfSupervisedClassifier = field(default=None, init=False)
    train_loader: DataLoader = field(default=None, init=False)
    valid_loader: DataLoader = field(default=None, init=False)

    def __post_init__(self):
        AuxiliaryTask.input_shape   = self.dataset.x_shape
        AuxiliaryTask.hidden_size   = self.hparams.hidden_size
        self.model = SelfSupervisedClassifier(
            input_shape=self.dataset.x_shape,
            num_classes=self.dataset.y_shape[0],
            hparams=self.hparams,
            config=self.config,
            tasks=self.hparams.get_tasks(),
        )
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


    def train_epoch(self, epoch: int):
        model = self.model
        dataloader = self.train_loader

        model.train()
        train_loss = 0.

        total_samples = len(dataloader.dataset)

        total_aux_losses: Dict[str, float] = defaultdict(float)
        pbar = tqdm(total=total_samples)
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(model.device)
            batch_size = data.shape[0]

            model.optimizer.zero_grad()
            loss, losses = model.get_loss(data, target)
            loss.backward()
            train_loss += loss.item()
            model.optimizer.step()

            for loss_name, loss_tensor in losses.items():
                total_aux_losses[loss_name] += loss_tensor.item()

            if batch_idx % model.config.log_interval == 0:
                samples_seen = batch_idx * len(data)
                percent_done = 100. * batch_idx/ len(dataloader)
                average_loss_in_batch = loss.item() / len(data)

                message: Dict[str, Any] = OrderedDict()
                pbar.set_description(f"Epoch {epoch}")
                message["Average Total Loss"] = average_loss_in_batch


                for loss_name, loss_tensor in losses.items():
                    if loss_name.endswith("_scaled"):
                        continue
                    scaled_loss_tensor = losses.get(f"{loss_name}_scaled")
                    if scaled_loss_tensor is not None:
                        message[loss_name] = f"{loss_str(scaled_loss_tensor)} ({loss_str(loss_tensor)})"
                    else:
                        message[loss_name] = loss_str(loss_tensor)
                pbar.set_postfix(message)
            
            pbar.update(batch_size)

        average_loss = train_loss / total_samples
        pbar.close()
        print(f"====> Epoch: {epoch} Average total loss: {average_loss:.4f}", end="\r")
        return average_loss


    def test_epoch(self, epoch: int):
        model = self.model
        dataloader = self.valid_loader
        model.eval()
        test_loss = 0.

        total_aux_losses: Dict[str, float] = defaultdict(float)

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(model.device)
                total_loss, losses = model.get_loss(data, target)
                test_loss += total_loss

                for loss_name, loss_tensor in losses.items():
                    total_aux_losses[loss_name] += loss_tensor.item()

                if i == 0:
                    n = min(data.size(0), 8)
                    fake = model.reconstruct(data)
                    # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
                    comparison = torch.cat([data[:n], fake[:n]])
                    save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)

        test_loss /= len(dataloader.dataset)
        print(f"====> Test set loss: {test_loss:.4f}")
        print(*[f"{loss_name}: {loss_str(loss_tensor)}" for loss_name, loss_tensor in losses.items()], sep=" ")
