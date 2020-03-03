import os
import pprint
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import simple_parsing
import torch
import torch.utils.data
import tqdm
from simple_parsing import (ArgumentParser, choice, field, list_field,
                            subparsers)
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common.losses import LossInfo
from config import Config
from datasets.bases import Dataset
from datasets.mnist import Mnist
from experiments.baseline import Baseline
from experiments.experiment import Experiment
from models.ss_classifier import SelfSupervisedClassifier
from tasks import AuxiliaryTask, VAEReconstructionTask
from tasks.torchvision.adjust_brightness import AdjustBrightnessTask
from utils.logging import loss_str


@dataclass
class SelfSupervised(Baseline):
    """ Simply adds auxiliary tasks to the IID experiment. """
    hparams: SelfSupervisedClassifier.HParams = SelfSupervisedClassifier.HParams(detach_classifier=False)
    
    def __post_init__(self):
        super().__post_init__()

        self.reconstruction_task: Optional[VAEReconstructionTask] = None
        # find the reconstruction task, if there is one.
        for aux_task in self.model.tasks:
            if isinstance(aux_task, VAEReconstructionTask):
                self.reconstruction_task = aux_task
                break

    def test_iter(self, epoch: int, dataloader: DataLoader):
        yield from super().test_iter(epoch, dataloader)
        if self.reconstruction_task:
            with torch.no_grad():
                sample = self.reconstruction_task.generate(torch.randn(64, self.hparams.hidden_size))
                sample = sample.cpu().view(64, 1, 28, 28)
                save_image(sample, os.path.join(self.config.log_dir, f"sample_{epoch}.png"))
        
    def make_plots(self, train_epoch_loss: List[LossInfo], valid_epoch_loss: List[LossInfo]):
        # TODO: make plots that are specific to self-supervised context?
        super().make_plots(train_epoch_loss, valid_epoch_loss)
    
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

# if i == 0:
            #     n = min(data.size(0), 8)
            #     fake = self.reconstruction_task.reconstruct(data)
            #     # fake = recon_batch.view(model.hparams.batch_size, 1, 28, 28)
            #     comparison = torch.cat([data[:n], fake[:n]])
            #     save_image(comparison.cpu(), f"results/reconstruction_{epoch}.png", nrow=n)
