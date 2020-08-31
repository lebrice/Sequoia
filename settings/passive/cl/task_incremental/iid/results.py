from dataclasses import dataclass
from typing import Dict, List

from common import Metrics, Loss

from .. import TaskIncrementalResults


@dataclass
class IIDResults(TaskIncrementalResults):
    pass