
from .hopper import HopperEnv, ContinualHopperEnv
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from typing import ClassVar, Type

# TODO: There is a bug in the way the hopper XML is generated, where the sticks / joints don't seem to follow. 
bob = ContinualHopperEnv(body_name_to_size_scale={"torso": 2})
assert False, bob

class TestContinualHopperEnv(ModifiedGravityEnvTests, ModifiedSizeEnvTests):
    Environment: ClassVar[Type[ContinualHopperEnv]] = ContinualHopperEnv
