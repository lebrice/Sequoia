import pytest
from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings import ClassIncrementalSetting

from .multihead_classifier import ExampleTaskInferenceMethod


@pytest.mark.timeout(300)
def test_SL_track(sl_track_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the Setting of the sl track of the competition.
    """
    method = ExampleTaskInferenceMethod(
        hparams=ExampleTaskInferenceMethod.HParams(max_epochs_per_task=1)
    )
    results = sl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    assert 0.30 <= online_perf.objective <= 0.50
    final_perf = results.average_final_performance
    assert 0.02 <= final_perf.objective <= 0.05
