from .bases import AuxiliaryTask, TaskType
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction import VAEReconstructionTask
from .rotation import RotationTask
from .tasks import AuxiliaryTaskOptions
from .torchvision.adjust_brightness import AdjustBrightnessTask
from .torchvision.transformation import (ClassifyTransformationTask,
                                         RegressTransformationTask)

__all__ = [
    "AuxiliaryTask", "TaskType",
    "IrmTask",
    "JigsawPuzzleTask"
    "ManifoldMixupTask", "MixupTask"
    "PatchLocationTask"
    "VAEReconstructionTask"
    "RotationTask",
    "AuxiliaryTaskOptions",
    "AdjustBrightnessTask",
    "ClassifyTransformationTask", "RegressTransformationTask",
]
