import enum
from dataclasses import dataclass
from typing import Dict, NewType, Tuple, Union, cast, ClassVar

from simple_parsing import mutable_field
from torch import nn

from .auxiliary_task import AuxiliaryTask
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction.vae import VAEReconstructionTask
from .transformation_based import (AdjustBrightnessTask,
                                   ClassifyTransformationTask,
                                   RegressTransformationTask, RotationTask)
from .simclr.simclr_task import SimCLRTask

class Tasks:
    """Enum-like class that just holds the names of each task.

    NOTE: Not using enum.Enum since it's a bit annoying to have to do
    Tasks.SUPERVISED.value instead of Tasks.SUPERVISED to get a str. 
    """
    SUPERVISED: ClassVar[str] = "supervised"
    RECONSTRUCTION: ClassVar[str] = "reconstruction"
    MIXUP: ClassVar[str] = "mixup"
    MANIFOLD_MIXUP: ClassVar[str] = "manifold_mixup"
    ROTATION: ClassVar[str] = "rotation"
    JIGSAW: ClassVar[str] = "jigsaw"
    IRM: ClassVar[str] = "irm"
    BRIGHTNESS: ClassVar[str] = "adjust_brightness"
    SIMCLR: ClassVar[str] = "simclr"


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
                    hidden_size: int) -> Dict[str, AuxiliaryTask]:
        tasks = nn.ModuleDict()
        if self.reconstruction:
            tasks[Tasks.RECONSTRUCTION] = VAEReconstructionTask(options=self.reconstruction)
        if self.mixup:
            tasks[Tasks.MIXUP] = MixupTask(options=self.mixup)
        if self.manifold_mixup:
            tasks[Tasks.MANIFOLD_MIXUP] = ManifoldMixupTask(options=self.manifold_mixup)
        if self.rotation:
            tasks[Tasks.ROTATION] = RotationTask(options=self.rotation)
        if self.jigsaw:
            tasks[Tasks.JIGSAW] = JigsawPuzzleTask(options=self.jigsaw)
        if self.irm:
            tasks[Tasks.IRM] = IrmTask(options=self.irm)
        if self.brightness:
            tasks[Tasks.BRIGHTNESS] = AdjustBrightnessTask(options=self.brightness)
        if self.simclr:
            tasks[Tasks.SIMCLR] = SimCLRTask(options=self.simclr)
        return cast(Dict[str, AuxiliaryTask], tasks)
        return tasks
