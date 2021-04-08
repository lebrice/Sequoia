import pytest
from sequoia.common.config import Config
from sequoia.settings.active import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)
from sequoia.settings import Setting

from .sac import SACMethod
from typing import Type


@pytest.mark.parametrize(
    "Setting", [ContinualRLSetting, IncrementalRLSetting, TaskIncrementalRLSetting]
)
@pytest.mark.parametrize("observe_state", [True, False])
def test_continuous_mountaincar_state(Setting: Type[Setting], observe_state: bool):
    method = SACMethod()
    setting = Setting(
        dataset="MountainCarContinuous-v0",
        observe_state_directly=True,
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: ContinualRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())
