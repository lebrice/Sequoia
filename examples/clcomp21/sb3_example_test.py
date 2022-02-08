import pytest

from sequoia.client.setting_proxy import SettingProxy
from sequoia.conftest import slow
from sequoia.settings.rl import IncrementalRLSetting, RLSetting
from sequoia.settings.sl import ClassIncrementalSetting

from .sb3_example import CustomPPOMethod, CustomPPOModel


@pytest.mark.timeout(120)
def test_cartpole_state(cartpole_state_setting: SettingProxy[RLSetting]):
    """Applies this Method to a simple cartpole-state setting."""
    method = CustomPPOMethod(hparams=CustomPPOModel.HParams(n_steps=64))
    results = cartpole_state_setting.apply(method)
    assert results.to_log_dict()

    results: RLSetting.Results
    # TODO: BUG: The SB3 method uses more than the number of steps allowed, probably
    # while filling up its buffer.
    assert 150 < results.average_final_performance.mean_episode_length


@pytest.mark.timeout(120)
def test_incremental_cartpole_state(
    incremental_cartpole_state_setting: SettingProxy[IncrementalRLSetting],
):
    """Applies this Method to the class-incremental mnist Setting."""
    method = CustomPPOMethod()
    results = incremental_cartpole_state_setting.apply(method)
    assert results.to_log_dict()

    results: ClassIncrementalSetting.Results
    # TODO: Increase this bound
    assert 5 <= results.average_online_performance.objective
    assert 5 <= results.average_final_performance.objective


@pytest.mark.timeout(300)
def test_RL_track(rl_track_setting: SettingProxy[IncrementalRLSetting]):
    """Applies this Method to the Setting of the sl track of the competition."""
    method = CustomPPOMethod()
    results = rl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    # TODO: get an estimate of the upper bound of the random method on the RL track.
    assert 0 < online_perf.objective
    final_perf = results.average_final_performance
    assert 0 < final_perf.objective
