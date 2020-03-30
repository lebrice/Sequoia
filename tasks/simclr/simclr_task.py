from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchvision.transforms import Compose, ToPILImage, ToTensor, Lambda
from torchvision.transforms.functional import to_tensor
from common.losses import LossInfo
from common.metrics import get_metrics, RegressionMetrics
from tasks.auxiliary_task import AuxiliaryTask

try:
    from .falr.config import HParams
    from .falr.data import SimCLRAugment
    from .falr.experiment import nt_xent_loss
    from .falr.models import Projector
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()


class SimCLRTask(AuxiliaryTask):

    @dataclass
    class Options(AuxiliaryTask.Options, HParams):
        pass

    def __init__(self, name: str="SimCLR", options: "SimCLRTask.Options"=None):
        super().__init__(name=name, options=options)
        self.options: SimCLRTask.Options

        # Set the same values for equivalent hyperparameters
        self.options.image_size = AuxiliaryTask.input_shape[-1]
        self.options.double_augmentation = True
        self.options.repr_dim = AuxiliaryTask.hidden_size

        self.augment = Compose([
            ToPILImage(),
            SimCLRAugment(self.options),
            Lambda(lambda tup: list(map(to_tensor, tup)))
        ])
        self.projector = Projector(self.options)

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        # TODO: is there a more efficient way to do this than with map? (torch multiprocessing-ish?)
        x_augment = list(map(self.augment, x.cpu()))  # [2, B, C, H, W]
        x1 = torch.stack([pair[0] for pair in x_augment])  # [B, C, H, W]
        x2 = torch.stack([pair[1] for pair in x_augment])  # [B, C, H, W]
        
        h1 = self.encode(x1.to(self.device)).flatten(start_dim=1)  # [B, repr_dim]
        h2 = self.encode(x2.to(self.device)).flatten(start_dim=1)  # [B, repr_dim]
        
        z1 = self.projector(h1)  # [B, proj_dim]
        z2 = self.projector(h2)  # [B, proj_dim]
        z = torch.cat([z1, z2], dim=0)  # [B*2, proj_dim]

        loss = nt_xent_loss(z, self.options.xent_temp)
        
        return LossInfo(name=self.name, total_loss=loss)

