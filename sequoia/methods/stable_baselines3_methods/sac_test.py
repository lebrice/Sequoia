import pytest
from sequoia.common.config import Config
from sequoia.settings.active import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)
from sequoia.settings import Setting
from sequoia.conftest import slow

from .sac import SACMethod
from typing import Type


# TODO: Look into why SAC is so slow, there's probably a parameter which isn't being set
# properly.
@slow
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "Setting", [ContinualRLSetting, IncrementalRLSetting, TaskIncrementalRLSetting]
)
@pytest.mark.parametrize("observe_state", [True, False])
def test_continuous_mountaincar(Setting: Type[Setting], observe_state: bool):
    method = SACMethod()
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
