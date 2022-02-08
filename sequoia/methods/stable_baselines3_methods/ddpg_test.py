from typing import ClassVar, Type

import pytest

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import ContinuousActionSpaceMethodTests
from .ddpg import DDPGMethod, DDPGModel


@pytest.mark.timeout(60)
class TestDDPG(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = DDPGMethod
    Model: ClassVar[Type[BaseAlgorithm]] = DDPGModel
