from typing import ClassVar, Type

from sequoia.conftest import mujoco_required

from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from .modified_size_test import ModifiedSizeEnvTests
from .walker2d import ContinualWalker2dV2Env, ContinualWalker2dV3Env

pytestmark = mujoco_required


class TestContinualWalker2dV2Env(
    ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests
):
    Environment: ClassVar[Type[ContinualWalker2dV2Env]] = ContinualWalker2dV2Env


class TestContinualWalker2dV3Env(
    ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests
):
    Environment: ClassVar[Type[ContinualWalker2dV3Env]] = ContinualWalker2dV3Env
