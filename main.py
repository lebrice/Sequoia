from dataclasses import dataclass
from typing import Tuple

import simple_parsing
import torch
import torch.utils.data
from simple_parsing import ArgumentParser, field
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.bases import Model
from models.semi_supervised.classifier import SelfSupervisedClassifier, Options

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")

args = parser.parse_args()
options: Options = args.options
print("Options:", options)


model = SelfSupervisedClassifier(options, num_classes=10).to(options.device)
train_loader, test_loader = options.get_dataloaders()


def train_epoch(model: SelfSupervisedClassifier, dataloader: DataLoader, epoch: int):
    model.train()
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(model.device)
        
        model.optimizer.zero_grad()
        loss = model.get_loss(data, target)
        loss.backward()
        train_loss += loss.item()
        model.optimizer.step()

        if batch_idx % model.options.log_interval == 0:
            samples_seen = batch_idx * len(data)
            total_samples = len(dataloader.dataset)
            percent_done = 100. * batch_idx/ len(dataloader)
            average_loss_in_batch = loss.item() / len(data)
            print(f"Train Epoch: {epoch}",
                    f"[{samples_seen}/{total_samples}]",
                    f"Loss: {average_loss_in_batch:.6f}", sep="\t", end="\r")

    average_loss = train_loss / total_samples
    print(f"====> Epoch: {epoch} Average loss: {average_loss:.4f}", end="\r")
    return average_loss


def test_epoch(model: SelfSupervisedClassifier, dataloader: DataLoader, epoch: int):
    model.eval()
    test_loss = torch.zeros(1)
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data = data.to(model.device)
            test_loss += model.get_loss(data, target)
            if i == 0:
                n = min(data.size(0), 8)
                recon_batch = model.reconstruct(data)
                fake = recon_batch.view(model.options.batch_size, 1, 28, 28)
                comparison = torch.cat([data[:n], fake[:n]])
                save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)

    test_loss /= len(dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(options.epochs):
        train_epoch(model, train_loader, epoch)
        test_epoch(model, test_loader, epoch)
        with torch.no_grad():
            sample = model.generate(torch.randn(64, 20)).cpu().view(64, 1, 28, 28)
            save_image(sample, 'results/sample_' + str(epoch) + '.png')
