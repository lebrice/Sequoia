from functools import partial
from typing import ClassVar, Type

import numpy as np
import pytest
from continuum.scenarios import ClassIncremental
from sequoia.common.config import Config
from sequoia.common.metrics import ClassificationMetrics
from sequoia.common.spaces import Image
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.sl.environment import Environment
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from ..continual.environment_test import (
    TestContinualSLTestEnvironment as ContinualSLTestEnvironmentTests,
)
from ..continual.environment_test import TestEnvironment
from .environment import IncrementalSLEnvironment, IncrementalSLTestEnvironment
from .results import IncrementalSLResults
from sequoia.settings.assumptions.discrete_results import TaskSequenceResults


class TestIncrementalSLTestEnvironment(ContinualSLTestEnvironmentTests):
    Environment: ClassVar[Type[Environment]] = IncrementalSLEnvironment
    TestEnvironment: ClassVar[Type[TestEnvironment]] = partial(
        IncrementalSLTestEnvironment, task_schedule={i * 20: {} for i in range(5)}
    )

    def validate_results(self, results: TaskSequenceResults):
        # NOTE: We're not checking that the results here represent the entire transfer
        # matrix, because the test env is only used for one test loop.
        # The Setting creates the transfer matrix using multiple of these
        # `TaskSequenceResults` objects, each of which is obtained after training on
        # a task in the training loop.
        assert isinstance(results, TaskSequenceResults)
        assert isinstance(results.average_metrics, ClassificationMetrics)
        assert results.objective > 0
        # TODO: Fix this check:
        assert results.average_metrics.n_samples in [95, 100]
