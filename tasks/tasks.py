from dataclasses import dataclass
from torch import nn
from simple_parsing import mutable_field
from typing import Tuple, NewType, Dict, Union

from .bases import AuxiliaryTask, TaskType
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction import VAEReconstructionTask
from .rotation import RotationTask
from .torchvision.adjust_brightness import AdjustBrightnessTask
from .torchvision.transformation import (ClassifyTransformationTask,
                                         RegressTransformationTask)


@dataclass
class AuxiliaryTaskOptions:
    """ Options related to the auxiliary tasks.
    
    The "coefficient" parameter can be used to turn auxiliary tasks on or off.

    """
    reconstruction: VAEReconstructionTask.Options = mutable_field(VAEReconstructionTask.Options, coefficient=0.)
    mixup:          MixupTask.Options             = mutable_field(MixupTask.Options, coefficient=0.)
    manifold_mixup: ManifoldMixupTask.Options     = mutable_field(ManifoldMixupTask.Options, coefficient=0.)
    rotation:       RotationTask.Options          = mutable_field(RotationTask.Options, coefficient=0.)
    jigsaw:         JigsawPuzzleTask.Options      = mutable_field(JigsawPuzzleTask.Options, coefficient=0.)
    irm:            IrmTask.Options               = mutable_field(IrmTask.Options, coefficient=0.)
    brightness:     AdjustBrightnessTask.Options  = mutable_field(AdjustBrightnessTask.Options, coefficient=0.)

    def create_tasks(self,
                    input_shape: Tuple[int, ...],
                    hidden_size: int) -> nn.ModuleDict:
        tasks = nn.ModuleDict()
        if self.reconstruction:
            tasks["reconstruction"] = VAEReconstructionTask(options=self.reconstruction)
        if self.mixup:
            tasks["mixup"] = MixupTask(options=self.mixup)
        if self.manifold_mixup:
            tasks["manifold_mixup"] = ManifoldMixupTask(options=self.manifold_mixup)
        if self.rotation:
            tasks["rotation"] = RotationTask(options=self.rotation)
        if self.jigsaw:
            tasks["jigsaw"] = JigsawPuzzleTask(options=self.jigsaw)
        if self.irm:
            tasks["irm"] = IrmTask(options=self.irm)
        if self.brightness:
            tasks["adjust_brightness"] = AdjustBrightnessTask(options=self.brightness)
        return tasks
