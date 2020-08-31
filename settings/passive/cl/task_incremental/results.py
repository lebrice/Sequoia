import itertools
from dataclasses import dataclass
from typing import Dict, List

from common import ClassificationMetrics, Loss, Metrics, RegressionMetrics
from settings.passive.cl import ClassIncrementalResults
from utils import get_logger

logger = get_logger(__file__)

@dataclass
class TaskIncrementalResults(ClassIncrementalResults):
    pass