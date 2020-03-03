from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Type, Iterable, Any, List
from typing import ClassVar

import torch
import tqdm
from simple_parsing import field, choice, subparsers
from torch import nn, Tensor
from torch.utils.data import DataLoader

from common.losses import LossInfo
from common.metrics import Metrics
from config import Config
from datasets import Dataset
from datasets.mnist import Mnist
from models.classifier import Classifier
from tasks import AuxiliaryTask
from pathlib import Path
from models.ss_classifier import SelfSupervisedClassifier

@dataclass  # type: ignore
class Experiment:
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    hparams: Classifier.HParams = subparsers({
        "baseline": Classifier.HParams(),
        "self-supervised": SelfSupervisedClassifier.HParams(),
    })
    dataset: Dataset = choice({
        "mnist": Mnist(),
    }, default="mnist")
    name: str = "default"
    config: Config = Config()
    model: Classifier = field(default=None, init=False)
    
    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.        
        """
        AuxiliaryTask.input_shape   = self.dataset.x_shape
        self.model = self.get_model(self.dataset)
        self.config.log_dir /= self.name
        self.config.log_dir.mkdir(exist_ok=True)
    
    def load(self):
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders

    def get_model(self, dataset: Dataset) -> Classifier:
        if isinstance(dataset, Mnist):
            from models.ss_classifier import SelfSupervisedClassifier, MnistClassifier as SSMnistClassifier
            if isinstance(self.hparams, SelfSupervisedClassifier.HParams):
                return SSMnistClassifier(
                    hparams=self.hparams,
                    config=self.config,
                )
            else:
                from models.classifier import MnistClassifier
                return MnistClassifier(
                    hparams=self.hparams,
                    config=self.config,
                )
        raise NotImplementedError("TODO: add other datasets.")

    def run(self):
        self.load()

        self.model = self.model.to(self.config.device)

        train_epoch_losses: List[LossInfo] = []
        valid_epoch_losses: List[LossInfo] = []

        for epoch in range(self.hparams.epochs):
            train_batch_losses: List[LossInfo] = []
            valid_batch_losses: List[LossInfo] = []

            for train_loss in self.train_iter(epoch, self.train_loader):
                train_batch_losses.append(train_loss)
            train_epoch_losses.append(train_loss)
            
            for valid_loss in self.test_iter(epoch, self.valid_loader):
                valid_batch_losses.append(valid_loss)
            valid_epoch_losses.append(valid_loss)

            if self.config.wandb:
                # TODO: do some nice logging to wandb?:
                wandb.log(TODO)

            self.make_plots_for_epoch(epoch, train_batch_losses, valid_batch_losses)
        
        self.make_plots(train_epoch_losses, valid_epoch_losses)
    
    @abstractmethod
    def make_plots(self, train_epoch_loss: List[LossInfo], valid_epoch_loss: List[LossInfo]):
        pass

    @abstractmethod
    def make_plots_for_epoch(self, epoch: int, train_batch_losses: List[LossInfo], valid_batch_losses: List[LossInfo]):
        pass

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
