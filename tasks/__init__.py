from .auxiliary_task import AuxiliaryTask, TaskType
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction.vae import VAEReconstructionTask
from .tasks import AuxiliaryTaskOptions
from .transformation_based import AdjustBrightnessTask, RotationTask

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
