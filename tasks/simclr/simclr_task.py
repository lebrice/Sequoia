from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo
from common.metrics import RegressionMetrics, get_metrics
from tasks.auxiliary_task import AuxiliaryTask

try:
    from .falr.config import HParams
    from .falr.data import SimCLRAugment
    from .falr.losses import SimCLRLoss
    from .falr.models import Projector
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()


class SimCLRTask(AuxiliaryTask):

    @dataclass
    class Options(AuxiliaryTask.Options, HParams):
        """ Options for the SimCLR Task. """
        #simclr double augmentaiton
        double_augmentation: bool = True

    def __init__(self, name: str="SimCLR", options: "SimCLRTask.Options"=None):
        super().__init__(name=name, options=options)
        self.options: SimCLRTask.Options

        # Set the same values for equivalent hyperparameters
        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.double_augmentation = self.options.double_augmentation
        self.options.repr_dim = AuxiliaryTask.hidden_size

        self.augment = Compose([
            ToPILImage(),
            SimCLRAugment(self.options),
            Lambda(lambda tup: torch.stack([tup[0], tup[1]]))
        ])
        self.projector = Projector(self.options)
        self.i = 0
        self.loss = SimCLRLoss(self.options.proj_dim)

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        # TODO: is there a more efficient way to do this than with a list comprehension? (torch multiprocessing-ish?)
        # concat all the x's into a single list.
        x_t = torch.cat([self.augment(x_i) for x_i in x.cpu()], dim=0)   # [2*B, C, H, W]
        h_t = self.encode(x_t.to(self.device)).flatten(start_dim=1)  # [2*B, repr_dim]
        z = self.projector(h_t)  # [2*B, proj_dim]
        loss = self.loss(z, self.options.xent_temp)
        loss_info = LossInfo(name=self.name, total_loss=loss)
        return loss_info
