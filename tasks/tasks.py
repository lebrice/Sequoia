from dataclasses import dataclass
from typing import Dict, NewType, Tuple, Union

from simple_parsing import mutable_field
from torch import nn

from .auxiliary_task import AuxiliaryTask, TaskType
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction.vae import VAEReconstructionTask
from .transformation_based import (AdjustBrightnessTask,
                                   ClassifyTransformationTask,
                                   RegressTransformationTask, RotationTask)
from .simclr.simclr_task import SimCLRTask


@dataclass
class AuxiliaryTaskOptions:
    """ Options related to the auxiliary tasks.
    
    The "coefficient" parameter can be used to turn auxiliary tasks on or off.

    """
    reconstruction: VAEReconstructionTask.Options = mutable_field(VAEReconstructionTask.Options)
    mixup:          MixupTask.Options             = mutable_field(MixupTask.Options)
    manifold_mixup: ManifoldMixupTask.Options     = mutable_field(ManifoldMixupTask.Options)
    rotation:       RotationTask.Options          = mutable_field(RotationTask.Options)
    jigsaw:         JigsawPuzzleTask.Options      = mutable_field(JigsawPuzzleTask.Options)
    irm:            IrmTask.Options               = mutable_field(IrmTask.Options)
    brightness:     AdjustBrightnessTask.Options  = mutable_field(AdjustBrightnessTask.Options)
    simclr:         SimCLRTask.Options            = mutable_field(SimCLRTask.Options)

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
        if self.simclr:
            tasks["simclr"] = SimCLRTask(options=self.simclr)
        return tasks
