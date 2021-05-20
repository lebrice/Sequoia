
from .hopper import HopperEnv, ContinualHopperEnv
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from typing import ClassVar, Type


class TestContinualHopperEnv(ModifiedGravityEnvTests, ModifiedSizeEnvTests):
    Environment: ClassVar[Type[ContinualHopperEnv]] = ContinualHopperEnv
