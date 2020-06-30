from dataclasses import dataclass, field
from enum import Enum
from typing import List
from typing import Tuple, Union, Optional
import torch
from torch import Tensor
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from common.losses import LossInfo
from common.metrics import RegressionMetrics, get_metrics
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable
from tasks.auxiliary_task import AuxiliaryTask

try:
    from .falr.config import HParams, ExperimentType
    from .falr.data import SimCLRAugment
    from .falr.losses import SimCLRLoss
    from .falr.models import Projector
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()

class SimclrHParams(HParams, Serializable):
    pass



class SimCLRTask(AuxiliaryTask):

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        simclr_options: SimclrHParams = mutable_field(SimclrHParams)

    def __init__(self, name: str="SimCLR", options: "SimCLRTask.Options"=None):
        super().__init__(name=name, options=options)
        self.options: SimCLRTask.Options

        # Set the same values for equivalent hyperparameters
        self.hparams = self.options.simclr_options
        self.hparams.image_size = AuxiliaryTask.input_shape[-1]
        self.hparams.double_augmentation = True
        self.hparams.repr_dim = AuxiliaryTask.hidden_size

        self.augment = Compose([
            ToPILImage(),
            SimCLRAugment(self.hparams),
            Lambda(lambda tup: torch.stack([tup[0], tup[1]]))
        ])

        self.projector = Projector(self.hparams)
        self.i = 0
        self.loss = SimCLRLoss(self.hparams.proj_dim)

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        # TODO: is there a more efficient way to do this than with a list comprehension? (torch multiprocessing-ish?)
        # concat all the x's into a single list.
        x_t, _ = self.preprocess_simclr(x, None, self.hparams)
        #print(x_t.shape)
        del x
        del h_x
        #x_t = torch.cat([self.augment(x_i) for x_i in x.cpu()], dim=0)   # [2*B, C, H, W]
        h_t = self.encode(x_t).flatten(start_dim=1)  # [2*B, repr_dim]
        z = self.projector.to(h_t.device)(h_t)  # [2*B, proj_dim]
        loss = self.loss(z, self.hparams.xent_temp, device = z.device)
        loss_info = LossInfo(name=self.name, total_loss=loss)
        torch.cuda.empty_cache()
        return loss_info

    @staticmethod
    def preprocess_simclr(data:Tensor, target:Tensor=None, hparams=None) -> Tuple[Tensor, Optional[Tensor]]:
            x_device = data.device
            y_device = target.device if target is not None else None
            if hparams is None:
                options = SimCLRTask.Option
                # Set the same values for equivalent hyperparameters
                options.image_size = data.shape[-1]
                options.double_augmentation = True
                options.repr_dim = AuxiliaryTask.hidden_size
            else:
                options = hparams

            augment = Compose([
                ToPILImage(),
                SimCLRAugment(options),
                Lambda(lambda tup: torch.stack([tup[0], tup[1]]))
            ])
            data = data.cpu()
            data = torch.cat([augment(x_i) for x_i in data], dim=0)  # [2*B, C, H, W]
            target = torch.cat([ torch.stack([t,t]) for t in target.cpu()], dim=0).to(y_device) if target is not None else None
            return data.to(x_device), target