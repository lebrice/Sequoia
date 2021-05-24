from sequoia.conftest import mujoco_required
pytestmark = mujoco_required
from .walker2d import Walker2dEnv, ContinualWalker2dEnv
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from typing import ClassVar, Type


class TestContinualWalker2dEnv(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualWalker2dEnv]] = ContinualWalker2dEnv
