from typing import ClassVar, Type
import pytest

from .ddpg import DDPGMethod, DDPGModel
from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import ContinuousActionSpaceMethodTests


@pytest.mark.timeout(60)
class TestDDPG(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = DDPGMethod
    Model: ClassVar[Type[BaseAlgorithm]] = DDPGModel
    
