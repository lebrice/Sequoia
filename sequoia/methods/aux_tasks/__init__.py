from .auxiliary_task import AuxiliaryTask
from .ewc import EWCTask
from .reconstruction import AEReconstructionTask, VAEReconstructionTask
from .transformation_based import RotationTask

VAE: str = VAEReconstructionTask.name
AE: str = AEReconstructionTask.name
EWC: str = EWCTask.name
