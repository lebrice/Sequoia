from .bases import AuxiliaryTask
from .mixup import ManifoldMixupTask, MixupTask
from .patch_location import PatchLocationTask
from .patch_shuffling import PatchShufflingTask
from .reconstruction import VAEReconstructionTask
from .rotation import RotationTask

__all__ = [
    "AuxiliaryTask"    
    "ManifoldMixupTask", "MixupTask"
    "PatchLocationTask"
    "PatchShufflingTask"
    "VAEReconstructionTask"
    "RotationTask"
]