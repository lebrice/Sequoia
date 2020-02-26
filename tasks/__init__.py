from .bases import AuxiliaryTask, TaskType
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction import VAEReconstructionTask
from .rotation import RotationTask

__all__ = [
    "AuxiliaryTask", "TaskType",
    "IrmTask",
    "JigsawPuzzleTask"
    "ManifoldMixupTask", "MixupTask"
    "PatchLocationTask"
    "VAEReconstructionTask"
    "RotationTask"
]
