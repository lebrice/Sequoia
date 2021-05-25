import pytest
from sequoia.common.config import Config
from sequoia.settings.rl import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)
from sequoia.settings import Setting
from sequoia.conftest import slow

from .sac import SACMethod, SACModel
from typing import Type
from typing import ClassVar, Type

from .ddpg import DDPGMethod, DDPGModel
from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import ContinuousActionSpaceMethodTests


class TestSAC(ContinuousActionSpaceMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = SACMethod
    Model: ClassVar[Type[BaseAlgorithm]] = SACModel

    # TODO: Look into why SAC is so slow, there's probably a parameter which isn't being set
    # properly.
    @slow
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "Setting", [ContinualRLSetting, IncrementalRLSetting, TaskIncrementalRLSetting]
    )
    @pytest.mark.parametrize("observe_state", [True, False])
    def test_continuous_mountaincar(self, Setting: Type[Setting], observe_state: bool):
        method = self.Method()
        setting = Setting(
            dataset="MountainCarContinuous-v0",
            nb_tasks=2,
            steps_per_task=1_000,
            test_steps_per_task=1_000,
        )
        results: ContinualRLSetting.Results = setting.apply(
            method, config=Config(debug=True)
        )
        print(results.summary())
