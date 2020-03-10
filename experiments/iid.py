import os
import pprint
from collections import OrderedDict, defaultdict
from dataclasses import InitVar, asdict, dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type

import matplotlib.pyplot as plt
import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import ArgumentParser, choice, field, subparsers
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common.losses import LossInfo
from common.metrics import Metrics
from config import Config
from datasets.dataset import Dataset
from datasets.mnist import Mnist
from models.classifier import Classifier
from models.ss_classifier import SelfSupervisedClassifier
from tasks import AuxiliaryTask
from tasks.reconstruction import VAEReconstructionTask
from utils.logging import loss_str

from .experiment import Experiment


@dataclass
class IID(Experiment):
    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_task: Optional[VAEReconstructionTask] = None
        
        # If the model is a SelfSupervised classifier, it will have a `tasks` attribute.
        # find the reconstruction task, if there is one.
        if isinstance(self.model, SelfSupervisedClassifier):
            for aux_task in self.model.tasks:
                if isinstance(aux_task, VAEReconstructionTask):
                    self.reconstruction_task = aux_task
                    break
    
    def make_plots_for_epoch(self,
                             epoch: int,
                             train_losses: List[LossInfo],
                             valid_losses: List[LossInfo]) -> Dict[str, plt.Figure]:
        train_x: List[int] = list(accumulate([
            loss.metrics.n_samples for loss in train_losses
        ]))
        
        valid_loss: LossInfo = sum(valid_losses, LossInfo())
        valid_metrics = valid_loss.metrics
        
        accuracy = valid_metrics.accuracy
        class_accuracy = valid_metrics.class_accuracy
        if self.config.debug or self.config.verbose:
            print("Validation Loss:", valid_loss)

        fig: plt.Figure = plt.figure()
        ax1: plt.Axes = fig.add_subplot(1, 2, 1)
        ax1.set_title(f"Loss - Epoch {epoch}")
        ax1.set_xlabel("# of Samples")
        ax1.set_ylabel("Training Loss")
        
        # Plot the total loss
        total_train_losses = [loss.total_loss.cpu() for loss in train_losses]
        ax1.plot(train_x, total_train_losses, label="total loss")

        from utils.utils import to_dict_of_lists
        train_losses_dict = to_dict_of_lists([loss.losses for loss in train_losses])

        # # Plot all the other losses (auxiliary losses)
        # for loss_name, aux_losses in train_losses_dict.items():
        #     ax1.plot(train_x, aux_losses, label=loss_name)
        
        # add the vertical lines for task transitions (if any)
        for task_info in self.dataset.train_tasks:
            ax1.axvline(x=min(task_info.indices), color='r')

        ax1.legend(loc='upper right')
        
        n_classes = self.dataset.y_shape[0]
        classes = list(range(n_classes))
        ax2: plt.Axes = fig.add_subplot(1, 2, 2)
        
        ax2.bar(classes, class_accuracy)
        ax2.set_xlabel("Class")
        ax2.set_xticks(classes)
        ax2.set_xticklabels(classes)
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title(f"Validation Class Accuracy After Epoch {epoch}")
        fig.tight_layout()
        
        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=30)
        fig.savefig(self.plots_dir / f"epoch_{epoch}.jpg")
        
        return {"epoch_loss": fig}

    def make_plots(self, train_losses: List[LossInfo], valid_losses: List[LossInfo]) -> Dict[str, plt.Figure]:
        import matplotlib.pyplot as plt

        plots_dir = self.config.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        epochs = list(range(len(train_losses)))
        plots_dict: Dict[str, plt.Figure] = {}
              
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.subplots()
        
        ax.set_title("Total Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(epochs, [loss.total_loss for loss in train_losses], label="train")
        ax.plot(epochs, [loss.total_loss for loss in valid_losses], label="valid")
        ax.legend(loc='upper right')
        plots_dict["loss"] = fig

        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=30)
        fig.savefig(self.plots_dir / "loss.jpg")

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.plot(epochs, [loss.metrics.accuracy for loss in train_losses], label="train")
        ax.plot(epochs, [loss.metrics.accuracy for loss in valid_losses], label="valid")
        ax.legend(loc='lower right')
        
        plots_dict["accuracy"] = fig

        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(timeout=30)

        fig.savefig(self.plots_dir / "accuracy.jpg")
        return plots_dict


    def test_iter(self, epoch: int, dataloader: DataLoader):
        yield from super().test_iter(epoch, dataloader)
        if self.reconstruction_task:
            with torch.no_grad():
                reconstruction_images_dir = self.config.log_dir / "reconstruction"
                reconstruction_images_dir.mkdir(exist_ok=True)

                sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                save_image(sample, reconstruction_images_dir / f"sample_{epoch}.png")
    
    def train_iter(self, epoch: int, dataloader: DataLoader):
        yield from super().train_iter(epoch, dataloader)
        if self.reconstruction_task:
            with torch.no_grad():
                sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                save_image(sample, self.config.log_dir / f"generated_{epoch}.png")

                # os.makedirs(os.path.join(self.config.log_dir, "reconstruction"))
                # n = min(data.size(0), 8)
                # fake = self.reconstruction_task.reconstruct()
                # comparison = torch.cat([data[:n], fake[:n]])
                # save_image(comparison.cpu(), os.path.join(self.config.log_dir, "reconstruction", f"{epoch}.png"), nrow=n)

    def log_info(self, batch_loss_info: LossInfo, overall_loss_info: LossInfo) -> Dict:
        message = super().log_info(batch_loss_info, overall_loss_info)
        # add the logs for all the scaled losses:
        for loss_name, loss_tensor in batch_loss_info.losses.items():
            if loss_name.endswith("_scaled"):
                continue
            scaled_loss_tensor = batch_loss_info.losses.get(f"{loss_name}_scaled")
            if scaled_loss_tensor is not None:
                message[loss_name] = f"{loss_str(scaled_loss_tensor)} ({loss_str(loss_tensor)})"
            else:
                message[loss_name] = loss_str(loss_tensor)
        return message

