from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import simple_parsing
import torch
import torch.utils.data
from simple_parsing import ArgumentParser, field
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.bases import Model, Config
from models.semi_supervised.classifier import SelfSupervisedClassifier, HParams

parser = ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")

args = parser.parse_args()

import pprint

hparams: HParams = args.hparams
print("HyperParameters:")
pprint.pprint(asdict(hparams), indent=1)
config: Config = args.config
print("Config:")
pprint.pprint(asdict(config), indent=1)

print("=" * 40)
print("Starting training")

model = SelfSupervisedClassifier(hparams=hparams, config=config).to(config.device)
train_loader, test_loader = config.get_dataloaders()


def train_epoch(model: SelfSupervisedClassifier, dataloader: DataLoader, epoch: int):
    model.train()
    train_loss = 0.

    total_aux_losses: Dict[str, float] = defaultdict(float)

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(model.device)
        
        model.optimizer.zero_grad()
        loss, logs = model.get_loss(data, target)
        loss.backward()
        train_loss += loss.item()
        model.optimizer.step()

        for loss_name, loss_tensor in logs.items():
            total_aux_losses[loss_name] += loss_tensor.item()

        if batch_idx % model.config.log_interval == 0:
            samples_seen = batch_idx * len(data)
            total_samples = len(dataloader.dataset)
            percent_done = 100. * batch_idx/ len(dataloader)
            average_loss_in_batch = loss.item() / len(data)

            message: List[str] = []
            message.append(f"Train Epoch: {epoch}")
            message.append(f"[{samples_seen}/{total_samples}]")
            message.append(f"Average Total Loss: {average_loss_in_batch:.6f}")

            for loss_name, loss_tensor in logs.items():
                message.append(f"{loss_name}: ")
                loss = loss_tensor.item()
                if abs(loss) < 1e-3:
                    message.append(f"{loss:.2e}")
                else:
                    message.append(f"{loss:.3f}")
            print(*message, sep=" ", end="\r")

    average_loss = train_loss / total_samples
    print(f"====> Epoch: {epoch} Average loss: {average_loss:.4f}", end="\r")
    return average_loss


def test_epoch(model: SelfSupervisedClassifier, dataloader: DataLoader, epoch: int):
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data = data.to(model.device)
            test_loss += model.get_loss(data, target).item()
            if i == 0:
                n = min(data.size(0), 8)
                recon_batch = model.reconstruct(data)
                fake = recon_batch.view(model.options.batch_size, 1, 28, 28)
                comparison = torch.cat([data[:n], fake[:n]])
                save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)

    test_loss /= len(dataloader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")


if __name__ == "__main__":
    for epoch in range(hparams.epochs):
        train_epoch(model, train_loader, epoch)
        test_epoch(model, test_loader, epoch)
        with torch.no_grad():
            sample = model.generate(torch.randn(64, hparams.hidden_size))
            sample = sample.cpu().view(64, 1, 28, 28)
            save_image(sample, 'results/sample_' + str(epoch) + '.png')
