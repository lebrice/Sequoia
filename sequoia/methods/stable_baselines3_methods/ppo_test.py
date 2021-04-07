import pytest
from sequoia.common.config import Config
from sequoia.conftest import monsterkong_required
from sequoia.settings.active import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)

from .ppo import PPOMethod, PPOModel


def test_cartpole_state():
    # Steps seems to be batch_size * n_steps * n_epochs, but I might be wrong.
    method = PPOMethod(hparams=PPOModel.HParams(batch_size=10, n_steps=10, n_epochs=2))
    setting = IncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
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
    method = PPOMethod(hparams=PPOModel.HParams(batch_size=10, n_steps=10, n_epochs=2))
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

