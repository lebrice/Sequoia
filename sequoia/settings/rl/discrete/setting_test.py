from ..continual.setting_test import TestContinualRLSetting as ContinualRLSettingTests
from .setting import DiscreteTaskAgnosticRLSetting
from typing import ClassVar, Type 
from sequoia.settings import Setting
from sequoia.settings.rl.envs import ATARI_PY_INSTALLED, MONSTERKONG_INSTALLED, MUJOCO_INSTALLED


class TestDiscreteTaskAgnosticRLSetting(ContinualRLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticRLSetting

    # IDEA: Create a fixture that creates the Setting which can then be tested.
    # TODO: Maybe this is a bit too complicated..
    @pytest.fixture(
        params=[("CartPole-v0", False), ("CartPole-v0", True),]
        + (
            [
                # Since the AtariWrapper gets added by default
                # param_requires_atari_py("Breakout-v0", True, Image(0, 255, (84, 84, 1)),),
                ("Breakout-v0", False),
            ]
            if ATARI_PY_INSTALLED
            else []
        )
        + (
            [("MetaMonsterKong-v0", False), ("MetaMonsterKong-v0", True),]
            if MONSTERKONG_INSTALLED
            else []
        )
        + (
            [("HalfCheetah-v2", False), ("Hopper-v2", False), ("Walker2d-v2", False),]
            if MUJOCO_INSTALLED
            else []
            # TODO: Add support for duckytown envs!!
            # ("duckietown", (120, 160, 3)),
        ),
        scope="session",
    )
    def setting(self, request):
        dataset, force_pixel_observations = request.param
        setting = self.Setting(
            dataset=dataset, force_pixel_observations=force_pixel_observations,
        )

        yield setting
        # assert False, setting