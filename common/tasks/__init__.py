from .auxiliary_task import AuxiliaryTask
from .ewc import EWC
from .irm import IrmTask
from .jigsaw_puzzle import JigsawPuzzleTask
from .mixup import ManifoldMixupTask, MixupTask
from .reconstruction import AEReconstructionTask, VAEReconstructionTask
from .simclr import SimCLRTask
from .transformation_based import AdjustBrightnessTask, RotationTask

SIMCLR: str = SimCLRTask.name
VAE: str = VAEReconstructionTask.name
AE: str = AEReconstructionTask.name
