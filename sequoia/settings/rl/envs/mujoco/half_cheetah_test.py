from sequoia.conftest import mujoco_required
pytestmark = mujoco_required

from .half_cheetah import HalfCheetahEnv, ContinualHalfCheetahEnv
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from typing import ClassVar, Type, List

from sequoia.conftest import mujoco_required


@mujoco_required
class TestHalfCheetah(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualHalfCheetahEnv]] = ContinualHalfCheetahEnv
