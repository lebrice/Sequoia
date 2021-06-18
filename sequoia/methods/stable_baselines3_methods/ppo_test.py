from typing import ClassVar, Type

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import DiscreteActionSpaceMethodTests
from .ppo import PPOMethod, PPOModel


class TestPPO(DiscreteActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = PPOMethod
    Model: ClassVar[Type[BaseAlgorithm]] = PPOModel

