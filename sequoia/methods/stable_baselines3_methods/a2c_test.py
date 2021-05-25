from typing import ClassVar, Type

from .a2c import A2CMethod, A2CModel
from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import DiscreteActionSpaceMethodTests


class TestA2C(DiscreteActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = A2CMethod
    Model: ClassVar[Type[BaseAlgorithm]] = A2CModel
    
