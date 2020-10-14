from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import pl_bolts
from common.loss import Loss
from methods.models import BaselineModel
from methods.aux_tasks.reconstruction import AEReconstructionTask, VAEReconstructionTask
from utils.logging_utils import get_logger

logger = get_logger(__file__)


@dataclass
class SaveVaeSamplesCallback(Callback):
    """ Callback which saves some generated/reconstructed samples.
    
    Reconstructs and/or generates samples periodically during training if any of
    of the autoencoder/generative model based auxiliary tasks are used.
    """
    def __post_init__(self, *args, **kwargs):
        self.reconstruction_task: Optional[AEReconstructionTask] = None
        self.generation_task: Optional[VAEReconstructionTask] = None
        self.latents_batch: Optional[Tensor] = None
        self.model: Classifier
        self.trainer: Trainer
    
    def setup(self, trainer, pl_module, stage: str):
        """Called when fit or test begins"""
        super().setup(trainer, pl_module, stage)

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        self.trainer = trainer
        self.model = pl_module
        from methods.self_supervision import SelfSupervisedModel
        if isinstance(pl_module, SelfSupervisedModel):
            # if our model has auxiliary tasks (i.e., if it's a self-supervised model.)
            if VAEReconstructionTask.name in self.model.tasks:
                self.reconstruction_task = self.model.tasks[VAEReconstructionTask.name]
                self.generation_task = self.reconstruction_task
                self.latents_batch = torch.randn(64, self.model.hp.hidden_size)
            
            elif AEReconstructionTask.name in pl_module.tasks:
                self.reconstruction_task = self.model.tasks[AEReconstructionTask.name]
                self.generation_task = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaselineModel):
        # do something
        if self.generation_task:    
            # Save a batch of fake images after each epoch.
            self.generate_samples()
        
        ## Reconstruct some samples after each epoch.
        # TODO: change this to use an interval instead.
        x_batch = None
        if x_batch is not None:
            self.reconstruct_samples(x_batch)
    
    @torch.no_grad()
    def reconstruct_samples(self, data: Tensor):
        if not self.reconstruction_task or not self.reconstruction_task.enabled:
            return
        n = min(data.size(0), 16)
        
        originals = data[:n]
        reconstructed = self.reconstruction_task.reconstruct(originals)
        comparison = torch.cat([originals, reconstructed])

        reconstruction_images_dir = self.model.config.log_dir / "reconstruction"
        reconstruction_images_dir.mkdir(parents=True, exist_ok=True)
        file_name = reconstruction_images_dir / f"step_{self.trainer.global_step:08d}.png"
        comparison = comparison.cpu().detach()
        
        if self.trainer.logger:
            self.trainer.logger.log({"reconstruction": wandb.Image(comparison)})
        save_image(comparison, file_name, nrow=n)

    @torch.no_grad()
    def generate_samples(self):
        if not self.generation_task or not self.generation_task.enabled:
            return
        n = 64
        latents = self.latents_batch
        fake_samples = self.generation_task.generate(latents)
        fake_samples = fake_samples.cpu().reshape(n, *reversed(self.model.setting.dims))
        # fake_samples = (fake_samples * 255).astype(np.uint8)
        
        generation_images_dir = self.model.config.log_dir / "generated_samples"
        generation_images_dir.mkdir(parents=True, exist_ok=True)
        file_name = generation_images_dir / f"step_{self.trainer.global_step:08d}.png"

        if self.model.logger:
            self.model.logger.experiment.log({"generated": wandb.Image(fake_samples)})
        
        save_image(fake_samples, file_name, normalize=True)
        logger.debug(f"saved image at path {file_name}")
