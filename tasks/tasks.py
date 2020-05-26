import enum
from dataclasses import dataclass
from typing import ClassVar, Dict, NewType, Tuple, Union, cast

from simple_parsing import mutable_field
from torch import nn

from .auxiliary_task import AuxiliaryTask
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction.ae import AEReconstructionTask
from .reconstruction.vae import VAEReconstructionTask
from .simclr.simclr_task import SimCLRTask
from .transformation_based import (AdjustBrightnessTask,
                                   ClassifyTransformationTask,
                                   RegressTransformationTask, RotationTask)
from config import Config

logger = Config.get_logger(__file__)

class Tasks:
    """Enum-like class that just holds the names of each task.

    NOTE: Not using enum.Enum since it's a bit annoying to have to do
    Tasks.SUPERVISED.value instead of Tasks.SUPERVISED to get a str. 
    """
    SUPERVISED: ClassVar[str] = "supervised"
    VAE: ClassVar[str] = "vae"
    AE: ClassVar[str] = "ae"
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
    vae:            VAEReconstructionTask.Options = mutable_field(VAEReconstructionTask.Options)
    ae:             AEReconstructionTask.Options  = mutable_field(AEReconstructionTask.Options)
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
        if self.ae:
            tasks[Tasks.AE] = AEReconstructionTask(options=self.ae)
        if self.vae:
            tasks[Tasks.VAE] = VAEReconstructionTask(options=self.vae)
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
        
        for name, task in tasks.items():
            assert isinstance(task, AuxiliaryTask), f"Task {task} should be a subclass of {AuxiliaryTask}."
            if task.coefficient != 0:
                logger.info(f"enabling the '{name}' auxiliary task (coefficient of {task.coefficient})")
                task.enable()
        return cast(Dict[str, AuxiliaryTask], tasks)
