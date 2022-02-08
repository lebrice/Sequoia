from typing import ClassVar, Type

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import ContinuousActionSpaceMethodTests
from .td3 import TD3Method, TD3Model


class TestDDPG(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = TD3Method
    Model: ClassVar[Type[BaseAlgorithm]] = TD3Model
