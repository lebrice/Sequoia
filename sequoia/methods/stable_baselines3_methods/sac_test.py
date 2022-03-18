from typing import ClassVar, Type

import pytest

from sequoia.common.config import Config
from sequoia.conftest import slow
from sequoia.settings import Setting
from sequoia.settings.rl import ContinualRLSetting, IncrementalRLSetting, TaskIncrementalRLSetting

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import ContinuousActionSpaceMethodTests
from .sac import SACMethod, SACModel


@slow
@pytest.mark.timeout(120)
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
            train_steps_per_task=1_000,
            test_steps_per_task=1_000,
        )
        results: ContinualRLSetting.Results = setting.apply(method, config=Config(debug=True))
        print(results.summary())
