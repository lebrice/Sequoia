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

from models.base import Model
from models.semi_supervised.vae_classifier import VaeClassifier
from options import Options

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")

args = parser.parse_args()
options: Options = args.options
print(options)


model = VaeClassifier(num_classes=10).to(options.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader, test_loader = options.get_dataloaders()

def train_epoch(model: Model, dataloader: DataLoader, epoch: int, optimizer: optim.Optimizer, options: Options):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(options.device)
        
        optimizer.zero_grad()
        loss = model.get_loss(data, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % options.log_interval == 0:
            samples_seen = batch_idx * len(data)
            total_samples = len(dataloader.dataset)
            percent_done = 100. * batch_idx/ len(dataloader)
            average_loss_in_batch = loss.item() / len(data)
            print(f"Train Epoch: {epoch} [{samples_seen}/{total_samples} Loss: {average_loss_in_batch:.6f}")

    average_loss = train_loss / total_samples
    return average_loss
    print(f"====> Epoch: {epoch} Average loss: {average_loss:.4f}")


def test_epoch(model: Model, dataloader: DataLoader, epoch: int, options: Options):
    self.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data = data.to(options.device)
            test_loss += self.get_loss(data, target)
            if i == 0:
                n = min(data.size(0), 8)
                fake = recon_batch.view(options.batch_size, 1, 28, 28)
                comparison = torch.cat([data[:n], fake[:n]])
                save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)

    test_loss /= len(dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, options.epochs + 1):
        train_epoch(model, train_loader, epoch, optimizer, options)
        test_epoch(model, valid_loader, epoch, options)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(options.device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
