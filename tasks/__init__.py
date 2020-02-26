from .bases import AuxiliaryTask, TaskType
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .reconstruction import VAEReconstructionTask
from .rotation import RotationTask

__all__ = [
    "AuxiliaryTask", "TaskType",
    "JigsawPuzzleTask"
    "ManifoldMixupTask", "MixupTask"
    "PatchLocationTask"
    "VAEReconstructionTask"
    "RotationTask"
]
