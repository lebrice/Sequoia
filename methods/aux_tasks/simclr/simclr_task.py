"""SimCLR auxiliary task.

Currently uses the code from @nitarshan's (private) repo, which is added as a
submodule.
TODO: #5 Refactor the SimCLR Auxiliary Task to use pl_bolts's implementation
"""
from dataclasses import dataclass, field
from typing import ClassVar, Dict

import torch
from torch import Tensor
from torchvision.transforms import Compose, Lambda, ToPILImage

from common.loss import Loss
from simple_parsing import mutable_field
from simple_parsing.helpers import Serializable

from ..auxiliary_task import AuxiliaryTask

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
    name: ClassVar[str] = "simclr"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options for the SimCLR aux task. """
        # Hyperparameters from the falr submodule.
        simclr_options: SimclrHParams = mutable_field(SimclrHParams)

    def __init__(self, name: str="simclr", options: "SimCLRTask.Options"=None):
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

    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None) -> Loss:
        x = forward_pass["x"]
        # TODO: is there a more efficient way to do this than with a list
        # comprehension? (torch multiprocessing-ish?)
        # concat all the x's into a single list.
        x_t = torch.cat([self.augment(x_i) for x_i in x.cpu()], dim=0)   # [2*B, C, H, W]
        h_t = self.encode(x_t.to(self.device)).flatten(start_dim=1)  # [2*B, repr_dim]
        z = self.projector(h_t)  # [2*B, proj_dim]
        loss = self.loss(z, self.hparams.xent_temp)
        loss_object = Loss(name=self.name, loss=loss)
        return loss_object
