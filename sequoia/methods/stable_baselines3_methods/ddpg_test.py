from typing import ClassVar, Type

from .ddpg import DDPGMethod, DDPGModel
from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import ContinuousActionSpaceMethodTests


class TestDDPG(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = DDPGMethod
    Model: ClassVar[Type[BaseAlgorithm]] = DDPGModel
    
