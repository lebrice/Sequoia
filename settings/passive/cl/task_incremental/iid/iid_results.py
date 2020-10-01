"""Defines the Results of apply a Method to an IID Setting.  
"""

from dataclasses import dataclass
from typing import Dict, List

from common import Metrics, Loss

from .. import TaskIncrementalResults


@dataclass
class IIDResults(TaskIncrementalResults):
    """Results of applying a Method on an IID Setting.    
    TODO: This should be customized, as it doesn't really make sense to use the
    same plots as in ClassIncremental (there is only one task).
    """