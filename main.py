import pprint
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import simple_parsing
import torch
import torch.utils.data
from simple_parsing import ArgumentParser, field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.bases import Config, Model
from models.semi_supervised.classifier import HParams, SelfSupervisedClassifier

parser = ArgumentParser()
parser.add_arguments(HParams, dest="hparams")
parser.add_arguments(Config, dest="config")

args = parser.parse_args()


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
        loss, losses = model.get_loss(data, target)
        loss.backward()
        train_loss += loss.item()
        model.optimizer.step()

        for loss_name, loss_tensor in losses.items():
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

            def loss_str(loss_tensor: Tensor) -> str:
                loss = loss_tensor.item()
                if loss == 0:
                    return "0"
                elif abs(loss) < 1e-3 or abs(loss) > 1e3:
                    return f"{loss:.1e}"
                else:
                    return f"{loss:.3f}"

            for loss_name, loss_tensor in losses.items():
                if loss_name.endswith("_scaled"):
                    continue
                scaled_loss_tensor = losses.get(f"{loss_name}_scaled")
                if scaled_loss_tensor is not None:
                    message.append(f"{loss_name}: {loss_str(scaled_loss_tensor)} ({loss_str(loss_tensor)})")
                else:
                    message.append(f"{loss_name}: {loss_str(loss_tensor)}")
            print(*message, sep=" ", end="\r")

    average_loss = train_loss / total_samples
    print(f"====> Epoch: {epoch} Average total loss: {average_loss:.4f}", end="\r")
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
