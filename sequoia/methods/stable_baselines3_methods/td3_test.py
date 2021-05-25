from typing import ClassVar, Type

from .td3 import TD3Method, TD3Model
from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import ContinuousActionSpaceMethodTests


class TestDDPG(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = TD3Method
    Model: ClassVar[Type[BaseAlgorithm]] = TD3Model
