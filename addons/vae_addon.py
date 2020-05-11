from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from common.losses import LossInfo
from experiment import ExperimentBase
from tasks.reconstruction import AEReconstructionTask, VAEReconstructionTask
from tasks.tasks import Tasks
import wandb

@dataclass  # type: ignore
class ExperimentWithVAE(ExperimentBase):
    """ Add-on / mixin for Experiment which generates/reconstructs samples.
    
    Reconstructs and/or generates samples periodically during training if any of
    of the autoencoder/generative model based auxiliary tasks are used.
    """
    
    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_task: Optional[AEReconstructionTask] = None
        self.generation_task: Optional[VAEReconstructionTask] = None

    def init_model(self):
        self.model = super().init_model()
        # find the reconstruction task, if there is one.
        if Tasks.VAE in self.model.tasks:
            self.reconstruction_task = self.model.tasks[Tasks.VAE]
            self.generation_task = self.reconstruction_task
            self.latents_batch = torch.randn(64, self.hparams.hidden_size)
        elif Tasks.AE in self.model.tasks:
            self.reconstruction_task = self.model.tasks[Tasks.AE]
            self.generation_task = None
        return self.model

    def train_iter(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        x_batch: Optional[Tensor] = None
        for loss in super().train_iter(dataloader):
            x_batch = loss.tensors.get("x", x_batch)
            yield loss
        
        ## Reconstruct some samples after each epoch.
        # TODO: change this to use an interval instead.
        if x_batch is not None:
            self.reconstruct_samples(x_batch)
        self.generate_samples()
    
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
        comparison = comparison.cpu().detach()
        if self.config.use_wandb:
            self.log({"reconstruction": wandb.Image(comparison)})
        save_image(comparison, file_name, nrow=n)

    @torch.no_grad()
    def generate_samples(self):
        if not self.generation_task or not self.generation_task.enabled:
            return
        n = 64
        latents = torch.randn(64, self.hparams.hidden_size)
        fake_samples = self.generation_task.generate(latents)
        fake_samples = fake_samples.cpu().view(n, *self.dataset.x_shape)

        generation_images_dir = self.samples_dir / "generated_samples"
        generation_images_dir.mkdir(parents=True, exist_ok=True)
        file_name = generation_images_dir / f"step_{self.global_step:08d}.png"

        if self.config.use_wandb:
            self.log({"generated": wandb.Image(fake_samples)})
        save_image(fake_samples, file_name)
