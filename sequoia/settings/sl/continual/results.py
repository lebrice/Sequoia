from sequoia.settings.base import Results
from sequoia.settings.assumptions.continual import ContinualResults
from dataclasses import dataclass
from sequoia.common.metrics import MetricsType


class ContinualSLResults(ContinualResults[MetricsType]):
    pass