from typing import ClassVar, Type

from sequoia.conftest import mujoco_required

pytestmark = mujoco_required

from .half_cheetah import ContinualHalfCheetahV2Env, ContinualHalfCheetahV3Env
from .modified_gravity_test import ModifiedGravityEnvTests
from .modified_mass_test import ModifiedMassEnvTests
from .modified_size_test import ModifiedSizeEnvTests


@mujoco_required
class TestHalfCheetahV2(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualHalfCheetahV2Env]] = ContinualHalfCheetahV2Env


@mujoco_required
class TestHalfCheetahV3(ModifiedGravityEnvTests, ModifiedSizeEnvTests, ModifiedMassEnvTests):
    Environment: ClassVar[Type[ContinualHalfCheetahV3Env]] = ContinualHalfCheetahV3Env
