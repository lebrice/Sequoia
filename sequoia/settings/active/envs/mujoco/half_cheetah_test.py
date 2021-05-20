
from .half_cheetah import HalfCheetahEnv, ContinualHalfCheetahEnv
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from typing import ClassVar, Type, List


class TestHalfCheetah(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualHalfCheetahEnv]] = ContinualHalfCheetahEnv
    body_names: ClassVar[List[str]] = ["torso", "fthigh", "fshin", "ffoot"]
