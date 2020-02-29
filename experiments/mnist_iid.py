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
from models.classifier import HParams, SelfSupervisedClassifier, LossInfo
from utils.logging import loss_str

from experiments.experiment import Experiment
from config import Config
@dataclass
class MnistIID(Experiment):
    dataset: Mnist = Mnist(iid=True)
    hparams: HParams = HParams()
    config: Config = Config()

    model: SelfSupervisedClassifier = field(default=None, init=False)
    train_loader: DataLoader = field(default=None, init=False)
    valid_loader: DataLoader = field(default=None, init=False)

    def __post_init__(self):
        self.model = SelfSupervisedClassifier(
            input_shape=self.dataset.x_shape,
            num_classes=self.dataset.y_shape[0],
            hparams=self.hparams,
            config=self.config,
            tasks=[],
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
            loss_info = model.get_loss(data, target)
            loss = loss_info.total_loss
            losses = loss_info.losses
            tensors = loss_info.tensors
            
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
                loss_tuple = model.get_loss(data, target)
                test_loss += loss_tuple.total_loss

                for loss_name, loss_tensor in loss_tuple.losses.items():
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
