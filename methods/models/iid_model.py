from dataclasses import dataclass

from settings import IIDSetting
from utils import constant, get_logger

from .task_incremental_model import TaskIncrementalModel

logger = get_logger(__file__)


class IIDModel(TaskIncrementalModel[IIDSetting]):
    """ Model for an IID setting. 
    
    This is implemented quite simply a TaskIncrementalClassifier, but with only
    one train/val/test task.
    """
    
    @dataclass
    class HParams(TaskIncrementalModel.HParams):
        multihead: bool = constant(False)
