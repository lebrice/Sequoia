import pytest
from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings.passive import ClassIncrementalSetting
from sequoia.settings.active import IncrementalRLSetting

from .dummy_method import DummyMethod

@pytest.mark.timeout(300)
def test_SL_track(sl_track_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the Setting of the sl track of the competition.
    """
    method = DummyMethod()
    results = sl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    assert 0.02 <= online_perf.objective <= 0.05
    final_perf = results.average_final_performance
    assert 0.02 <= final_perf.objective <= 0.05


@pytest.mark.timeout(300)
def test_RL_track(rl_track_setting: SettingProxy[IncrementalRLSetting]):
    """ Applies this Method to the Setting of the sl track of the competition.
    """
    method = DummyMethod()
    results = rl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    # TODO: get an estimate of the upper bound of the random method on the RL track.
    TODO = 1_000  # this is way too large.
    assert 0 < online_perf.objective < TODO
    final_perf = results.average_final_performance
    assert 0 < final_perf.objective < TODO
