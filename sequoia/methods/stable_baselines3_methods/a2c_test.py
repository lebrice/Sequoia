import pytest
from sequoia.common.config import Config
from sequoia.conftest import monsterkong_required
from sequoia.settings.rl import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)

from .a2c import A2CMethod


def test_cartpole_state():
    method = A2CMethod()
    setting = IncrementalRLSetting(
        dataset="cartpole",
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())


@pytest.mark.timeout(120)
@monsterkong_required
def test_monsterkong():
    method = A2CMethod()
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())
