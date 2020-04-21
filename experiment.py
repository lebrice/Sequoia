import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, Generator, Iterable, List, Optional,
                    Tuple, Type, Union)

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from simple_parsing import choice, field, mutable_field, subparsers
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from common.losses import LossInfo
from common.metrics import (ClassificationMetrics, RegressionMetrics,
                            get_metrics)
from config import Config
from datasets import DatasetConfig
from datasets.cifar import Cifar10, Cifar100
from datasets.fashion_mnist import FashionMnist
from datasets.mnist import Mnist
from models.classifier import Classifier
from tasks import AuxiliaryTask, Tasks
from utils import utils
from utils.json_utils import is_json_serializable, to_str, to_str_dict
from utils.utils import add_prefix


@dataclass  # type: ignore
class Experiment:
    """ Describes the parameters of an experimental setting.
    
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize.

    TODO: Maybe add some code for saving/restoring experiments here?
    """
    # Model Hyper-parameters
    hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    
    dataset: DatasetConfig = choice({
        "mnist": Mnist(),
        "fashion_mnist": FashionMnist(),
        "cifar10": Cifar10(),
        "cifar100": Cifar100(),
    }, default="mnist")

    config: Config = Config()
    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None
    
    model: Classifier = field(default=None, init=False)

    def __post_init__(self):
        """ Called after __init__, used to initialize all missing fields.
        
        You can use this method to initialize the fields that aren't parsed from
        the command-line, such as `model`, etc.
        Additionally, the fields created here are not added in the wandb logs.       
        """
        AuxiliaryTask.input_shape   = self.dataset.x_shape

        # Set these shared attributes so that all the Auxiliary tasks can be created.
        if isinstance(self.dataset, (Mnist, FashionMnist)):
            AuxiliaryTask.input_shape = self.dataset.x_shape
            AuxiliaryTask.hidden_size = self.hparams.hidden_size


        self.train_loader: DataLoader = NotImplemented
        self.valid_loader: DataLoader = NotImplemented

        self.global_step: int = 0
        self.logger = logging.getLogger(__file__)
        if self.config.debug:
            self.logger.setLevel(logging.DEBUG)
        
        self._samples_dir: Optional[Path] = None
        self.reconstruction_task: Optional[VAEReconstructionTask] = None
        self.init_model()

        if self.notes:
            with open(self.log_dir / "notes.txt", "w") as f:
                f.write(self.notes)

    @abstractmethod
    def run(self):
        pass

    def load(self):
        """ Setup the dataloaders and other settings before training. """
        self.dataset.load(self.config)
        dataloaders = self.dataset.get_dataloaders(self.config, self.hparams.batch_size)
        self.train_loader, self.valid_loader = dataloaders
        self.global_step = 0

    def init_model(self):
        self.model = self.get_model_for_dataset(self.dataset).to(self.config.device)
        # find the reconstruction task, if there is one.
        if Tasks.RECONSTRUCTION in self.model.tasks:
            self.reconstruction_task = self.model.tasks[Tasks.RECONSTRUCTION]
            self.latents_batch = torch.randn(64, self.hparams.hidden_size)

    def get_model_for_dataset(self, dataset: DatasetConfig) -> Classifier:
        from models.mnist import MnistClassifier
        from models.cifar import Cifar10Classifier, Cifar100Classifier

        if isinstance(dataset, (Mnist, FashionMnist)):
            return MnistClassifier(hparams=self.hparams, config=self.config)
        elif isinstance(dataset, Cifar10):
            return Cifar10Classifier(hparams=self.hparams, config=self.config)
        elif isinstance(dataset, Cifar100):
            return Cifar100Classifier(hparams=self.hparams, config=self.config)
        else:
            raise NotImplementedError(f"TODO: add a model for dataset {dataset}.")

    def log(self, message: Union[str, Dict, LossInfo], value: Any=None, step: int=None, once: bool=False, prefix: str="", always_print: bool=False):
        if always_print or (self.config.debug and self.config.verbose):
            print(message, value if value is not None else "")

        with open(self.log_dir / "log.txt", "a") as f:
            print(message, value, file=f)

        if self.config.use_wandb:
            # if we want to long once (like a final result, step should be None)
            # else, if not given, we use the global step.
            step = None if once else (step or self.global_step)
            
            if message is None:
                return
            if isinstance(message, dict):
                message_dict = message
            elif isinstance(message, LossInfo):
                message_dict = message.to_log_dict()
            elif isinstance(message, str) and value is not None:
                message_dict = {message: value}
            else:
                message_dict = message
            
            if prefix:
                message_dict = utils.add_prefix(message_dict, prefix)
            
            wandb.log(message_dict, step=step)

    def train_until_convergence(self, train_dataset: Dataset,
                                      valid_dataset: Dataset,
                                      max_epochs: int,
                                      description: str=None,
                                      patience: int=3) -> Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]:
        train_dataloader = self.get_dataloader(train_dataset)
        valid_dataloader = self.get_dataloader(valid_dataset)
        n_steps = len(train_dataloader)
        
        if self.config.debug_steps:
            from itertools import islice
            n_steps = self.config.debug_steps
            train_dataloader = islice(train_dataloader, 0, n_steps)

        train_losses: Dict[int, LossInfo] = OrderedDict()
        valid_losses: Dict[int, LossInfo] = OrderedDict()

        valid_loss_gen = self.valid_performance_generator(valid_dataset)
        
        best_valid_loss: Optional[float] = None
        counter = 0

        message: Dict[str, Any] = OrderedDict()
        for epoch in range(max_epochs):
            pbar = tqdm.tqdm(train_dataloader, total=n_steps)
            desc = description or "" 
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")
            
            for batch_idx, train_loss in enumerate(self.train_iter(pbar)):
                if batch_idx % self.config.log_interval == 0:
                    # get loss on a batch of validation data:
                    valid_loss = next(valid_loss_gen)
                    valid_losses[self.global_step] = valid_loss
                    train_losses[self.global_step] = train_loss
                    
                    add_messages_for_batch(valid_loss, message, "Valid ")
                    add_messages_for_batch(train_loss, message, "Train ")
                    pbar.set_postfix(message)

                    self.log(train_loss, prefix="Train ")
                    self.log(valid_loss, prefix="Valid ")
            
            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataset, description=val_desc)
            val_loss = val_loss_info.total_loss
            
            if best_valid_loss is None or val_loss.item() < best_valid_loss:
                counter = 0
                best_valid_loss = val_loss.item()
            else:
                counter += 1
                print(f"Validation Loss hasn't decreased over the last {counter} epochs.")
                if counter == patience:
                    print(f"Exiting at step {self.global_step}, as validation loss hasn't decreased over the last {patience} epochs.")
                    break
        return train_losses, valid_losses

    def valid_performance_generator(self, valid_dataset: Dataset) -> Iterable[LossInfo]:
        periodic_valid_dataloader = self.get_dataloader(valid_dataset)
        while True:
            for batch in periodic_valid_dataloader:
                data = batch[0].to(self.model.device)
                target = batch[1].to(self.model.device) if len(batch) == 2 else None
                yield self.test_batch(data, target)

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        for batch in dataloader:
            data = batch[0].to(self.model.device)
            target = batch[1].to(self.model.device) if len(batch) == 2 else None
            batch_size = data.shape[0]

            yield self.train_batch(data, target)

            self.global_step += batch_size
        
        ## Reconstruct some samples after each epoch.
        if self.reconstruction_task and self.reconstruction_task.enabled:
            # use the last batch of x's.
            x_batch = data
            if x_batch is not None:
                self.reconstruct_samples(x_batch)

    def train_batch(self, data: Tensor, target: Tensor) -> LossInfo:
        self.model.optimizer.zero_grad()

        batch_loss_info = self.model.get_loss(data, target)
        total_loss = batch_loss_info.total_loss
        total_loss.backward()
                
        self.model.optimizer.step()
        return batch_loss_info

    def test(self, dataset: Dataset, description: str=None, name: str="Test") -> LossInfo:
        dataloader = self.get_dataloader(dataset)
        pbar = tqdm.tqdm(dataloader)
        desc = (description or "Test Epoch")
        
        pbar.set_description(desc)
        total_loss = LossInfo(name)
        message: Dict[str, Any] = OrderedDict()

        for batch_idx, loss in enumerate(self.test_iter(pbar)):
            total_loss += loss

            if batch_idx % self.config.log_interval == 0:
                add_messages_for_batch(total_loss, message, prefix="Test ")
                pbar.set_postfix(message)

        return total_loss

    def test_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.eval()
        for batch in dataloader:
            data = batch[0].to(self.model.device)
            target = batch[1].to(self.model.device) if len(batch) == 2 else None
            yield self.test_batch(data, target)

        ## Generate some samples after each test/eval epoch.
        self.generate_samples()
    
    @torch.no_grad()
    def test_batch(self, data: Tensor, target: Tensor) -> LossInfo:
        return self.model.get_loss(data, target)

    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=self.config.use_cuda
        )
    
    @torch.no_grad()
    def reconstruct_samples(self, data: Tensor):
        if not self.reconstruction_task or not self.reconstruction_task.enabled:
            return
        n = min(data.size(0), 16)
        
        originals = data[:n]
        reconstructed = self.reconstruction_task.reconstruct(originals)
        comparison = torch.cat([originals, reconstructed])

        reconstruction_images_dir = self.samples_dir / "reconstruction"
        reconstruction_images_dir.mkdir(parents=True, exist_ok=True)
        file_name = reconstruction_images_dir / f"step_{self.global_step:08d}.png"
        save_image(comparison.cpu(), file_name, nrow=n)

    @torch.no_grad()
    def generate_samples(self):
        if not self.reconstruction_task or not self.reconstruction_task.enabled:
            return
        n = 64
        latents = torch.randn(64, self.hparams.hidden_size)
        fake_samples = self.reconstruction_task.generate(latents)
        fake_samples = fake_samples.cpu().view(n, *self.dataset.x_shape)

        generation_images_dir = self.samples_dir / "generated_samples"
        generation_images_dir.mkdir(parents=True, exist_ok=True)
        file_name = generation_images_dir / f"step_{self.global_step:08d}.png"
        save_image(fake_samples, file_name)


    def _folder(self, folder: Union[str, Path], create: bool=True):
        path = self.config.log_dir / folder
        if create and not path.is_dir():
            path.mkdir(parents=False)
        return path

    @property
    def plots_dir(self) -> Path:
        return self._folder("plots")

    @property
    def samples_dir(self) -> Path:
        if self._samples_dir:
            return self._samples_dir
        self._samples_dir = self._folder("samples")
        return self._samples_dir
    
    @samples_dir.setter
    def samples_dir(self, value: Path) -> Path:
        self._samples_dir = value

    @property
    def checkpoints_dir(self) -> Path:
        return self._folder("checkpoints")

    @property
    def log_dir(self) -> Path:
        # Accessing this property doesn't create the folder.
        return self._folder("", create=False)

    @property
    def results_dir(self) -> Path:
        return self._folder("results", create=False)

    @property
    def config_path(self) -> Path:
        return self.log_dir / "config.pt"

    @property
    def started(self) -> bool:
        return is_nonempty_dir(self.config.log_dir)

    @property
    def done(self) -> bool:
        """Returns wether or not the experiment is complete.
        
        Returns:
            bool: Wether the experiment is complete or not (wether the
            results_dir exists and contains files).
        """
        return self.started and is_nonempty_dir(self.results_dir)
    
    def save_to_results_dir(self, results: Dict[Union[str, Path], Any]):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for path, result in results.items():
            path = Path(path) if isinstance(path, str) else path
            if path.suffix in {".csv", ""}:
                array = result.detach().numpy() if isinstance(result, Tensor) else result
                np.savetxt(self.results_dir / path.with_suffix(".csv"), array, delimiter=",")
            elif path.suffix == ".json":
                with open(self.results_dir / path, "w") as f:
                    json.dump(result, f, indent="\t")

    def save(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoints_dir / "config.json", "w") as f:
            d = asdict(self)
            d = to_str(d)
            json.dump(d, f, indent=1)

        # with self.config_path.open("w") as f:
        #     torch.save(self, f)

    @classmethod
    def load_from_config(cls, config_path: Union[Path, str]):
        with open(config_path) as f:
            return torch.load(f)

def is_nonempty_dir(path: Path):
    path.is_dir() and len(list(path.iterdir())) > 0


def add_messages_for_batch(loss: LossInfo, message: Dict, prefix: str=""):
    new_message: Dict[str, Union[str, float]] = OrderedDict()
    new_message[f"{prefix}Loss"] = loss.total_loss.item()
    for name, loss_info in loss.losses.items():
        new_message[f"{prefix}{name} Loss"] = loss.total_loss.item()
        for metric_name, metrics in loss_info.metrics.items():
            if isinstance(metrics, ClassificationMetrics):
                new_message[f"{prefix}{name} Acc"] = f"{metrics.accuracy:.2%}"
            elif isinstance(metrics, RegressionMetrics):
                new_message[f"{prefix}{name} MSE"] = metrics.mse.item()
    message.update(new_message)