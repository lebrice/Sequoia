import pytest
from sequoia.client.setting_proxy import SettingProxy
from sequoia.conftest import slow
from sequoia.settings.passive import ClassIncrementalSetting

from .classifier import Classifier, ExampleMethod


@pytest.mark.timeout(120)
def test_mnist(mnist_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the class-incremental mnist Setting.
    """
    method = ExampleMethod(hparams=Classifier.HParams(max_epochs_per_task=1))
    results = mnist_setting.apply(method)
    assert results.to_log_dict()

    results: ClassIncrementalSetting.Results
    assert 0.80 <= results.average_online_performance.objective <= 1.00
    assert 0.10 <= results.average_final_performance.objective <= 0.30


@slow
@pytest.mark.timeout(300)
def test_SL_track(sl_track_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the Setting of the sl track of the competition.
    """
    method = ExampleMethod(hparams=Classifier.HParams(max_epochs_per_task=1))
    results = sl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    assert 0.15 <= online_perf.objective <= 0.30
    final_perf = results.average_final_performance
    assert 0.01 <= final_perf.objective <= 0.05
