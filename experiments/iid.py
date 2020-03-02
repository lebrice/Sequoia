import pprint
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import ArgumentParser, field, subparsers, choice
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.mnist import Mnist
from models.classifier import Classifier, LossInfo
from utils.logging import loss_str

from experiments.experiment import Experiment
from config import Config

from datasets.bases import Dataset
@dataclass
class IID(Experiment):
    dataset: Dataset = choice({"mnist": Mnist()}, default="mnist")
    hparams: Classifier.HParams = Classifier.HParams()
    config: Config = Config()

    model: Classifier = field(default=None, init=False)
    train_loader: DataLoader = field(default=None, init=False)
    valid_loader: DataLoader = field(default=None, init=False)

    def __post_init__(self):
        if isinstance(self.dataset, Mnist):    
            from models.classifier import MnistClassifier
            self.model = MnistClassifier(
                hparams=self.hparams,
                config=self.config,
            )
        else:
            raise NotImplementedError("TODO: add other datasets.")
        dataloaders = self.dataset.get_dataloaders(self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders

    def run(self):
        for epoch in range(self.hparams.epochs):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            # with torch.no_grad():
            #     sample = model.generate(torch.randn(64, hparams.hidden_size))
            #     sample = sample.cpu().view(64, 1, 28, 28)
            #     save_image(sample, 'results/sample_' + str(epoch) + '.png')

    def log_info(self, batch_loss_info: LossInfo, overall_loss_info: LossInfo) -> Dict:
        message: Dict[str, Any] = OrderedDict()
        # average_accuracy = (overall_loss_info.metrics.get("accuracy", 0) / (batch_idx + 1))
        n_samples = overall_loss_info.metrics.n_samples
        message["metrics:"] = overall_loss_info.metrics
        message["Average Total Loss"] = overall_loss_info.total_loss.item() / n_samples
        return message

    def train_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        batch_size = data.shape[0]

        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)
        loss    = batch_loss_info.total_loss
        losses  = batch_loss_info.losses
        tensors = batch_loss_info.tensors
        metrics = batch_loss_info.metrics

        # print(overall_loss_info.metrics)

        loss.backward()
        self.model.optimizer.step()
        return batch_loss_info

    def train_epoch(self, epoch: int):
        model = self.model
        dataloader = self.train_loader

        model.train()
        train_loss = 0.

        total_samples = len(dataloader.dataset)
        
        overall_loss_info = LossInfo()

        
        accuracies: List[float] = []

        pbar = tqdm(total=total_samples)
        pbar = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(model.device)
            target = target.to(model.device)

            batch_loss_info = self.train_batch(batch_idx, data, target)
            
            overall_loss_info += batch_loss_info

            if batch_idx % model.config.log_interval == 0:
                pbar.set_description(f"Epoch {epoch}")
                message = self.log_info(batch_loss_info, overall_loss_info)
                pbar.set_postfix(message)

                # show training accuracy over the epoch:
                accuracies.append(overall_loss_info.metrics.accuracy)
            
            batch_size = data.shape[0]

        import matplotlib.pyplot as plt

        plt.plot(accuracies)
        plt.show()

        average_loss = train_loss / total_samples
        pbar.close()
        print(f"====> Epoch: {epoch} Average training loss: {average_loss:.4f}", end="\r")
        return average_loss

    def test_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        loss_info = self.model.get_loss(data, target)
        # if i == 0:
        #     n = min(data.size(0), 8)
        #     fake = model.reconstruct(data)
        #     # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
        #     comparison = torch.cat([data[:n], fake[:n]])
        #     save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)
        return loss_info

    def test_epoch(self, epoch: int):
        model = self.model
        dataloader = self.valid_loader
        model.eval()
        test_loss = 0.

        total_aux_losses: Dict[str, float] = defaultdict(float)
        overall_loss_info = LossInfo()

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(model.device)
                target = target.to(model.device)

                batch_loss_info = self.test_batch(i, data, target)
                
                overall_loss_info += batch_loss_info
                
        print(f"====> Test set loss: {overall_loss_info.total_loss.item():.4f}")
        print(f"====> Test set Metrics:", overall_loss_info.metrics)
        return overall_loss_info
