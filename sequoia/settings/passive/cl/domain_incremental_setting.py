from dataclasses import dataclass
from sequoia.utils.utils import constant
from .class_incremental_setting import ClassIncrementalSetting



@dataclass
class DomainIncrementalSetting(ClassIncrementalSetting):
    relabel: bool = constant(True)
    
    