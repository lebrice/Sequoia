from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import *
from typing import ClassVar

import torch
import tqdm
from simple_parsing import field, choice
from torch import nn, Tensor
from torch.utils.data import DataLoader

from common.losses import LossInfo
from common.metrics import Metrics
from config import Config
from datasets import Dataset
from datasets.mnist import Mnist
from models.classifier import Classifier


@dataclass
class Experiment:
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    # Dataset and preprocessing settings.
    dataset: Dataset = choice({
        "mnist": Mnist(),
    }, default="mnist")
    # Model Hyperparameters 
    hparams: Classifier.HParams = Classifier.HParams()
    # Settings related to the experimental setup (cuda, log_dir, etc.).
    config: Config = Config()
        
    model_class: Type[Classifier] = field(default=Classifier, init=False)
    model: Classifier = field(default=None, init=False)

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.        
        """ 
        pass

    def run(self):
        raise NotImplementedError("Implement the 'run' method in a subclass.")

    def train_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        batch_size = data.shape[0]
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)

        total_loss = batch_loss_info.total_loss
        losses     = batch_loss_info.losses
        tensors    = batch_loss_info.tensors
        metrics    = batch_loss_info.metrics

        total_loss.backward()
        self.model.optimizer.step()
        return batch_loss_info

    def train_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        overall_loss_info = LossInfo()

        pbar = tqdm.tqdm(dataloader) # disable=not (self.config.verbose or self.config.debug)
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.model.device)
            target = target.to(self.model.device)

            batch_loss_info = self.train_batch(batch_idx, data, target)
            yield batch_loss_info

            overall_loss_info += batch_loss_info

            if batch_idx % self.config.log_interval == 0:
                pbar.set_description(f"Epoch {epoch}")
                message = self.log_info(batch_loss_info, overall_loss_info)
                pbar.set_postfix(message)

        print(f"====> Epoch: {epoch}, total loss: {overall_loss_info.total_loss}, metrics: {overall_loss_info.metrics}")
        return overall_loss_info

    def test_batch(self, batch_idx: int, data: Tensor, target: Tensor) -> LossInfo:
        loss_info = self.model.get_loss(data, target)
        # if i == 0:
        #     n = min(data.size(0), 8)
        #     fake = model.reconstruct(data)
        #     # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
        #     comparison = torch.cat([data[:n], fake[:n]])
        #     save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)
        return loss_info

    def test_iter(self, epoch: int, dataloader: DataLoader) -> Iterable[LossInfo]:
        model = self.model
        model.eval()
        test_loss = 0.

        total_aux_losses: Dict[str, float] = defaultdict(float)
        overall_loss_info = LossInfo()

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(model.device)
                target = target.to(model.device)

                batch_loss_info = self.test_batch(i, data, target)
                yield batch_loss_info
                overall_loss_info += batch_loss_info
                
        print(f"====> Test set loss: {overall_loss_info.total_loss.item():.4f}")
        print(f"====> Test set Metrics:", overall_loss_info.metrics)
        return overall_loss_info

    def log_info(self, batch_loss_info: LossInfo, overall_loss_info: LossInfo) -> Dict:
        message: Dict[str, Any] = OrderedDict()
        # average_accuracy = (overall_loss_info.metrics.get("accuracy", 0) / (batch_idx + 1))
        n_samples = overall_loss_info.metrics.n_samples
        message["metrics:"] = overall_loss_info.metrics
        message["Average Total Loss"] = overall_loss_info.total_loss.item() / n_samples
        return message
