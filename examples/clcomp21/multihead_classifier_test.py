import pytest
from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings import ClassIncrementalSetting

from .conftest import slow
from .multihead_classifier import ExampleTaskInferenceMethod, MultiHeadClassifier


@pytest.mark.timeout(120)
def test_mnist(mnist_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the class-incremental mnist Setting.
    """
    method = ExampleTaskInferenceMethod(
        hparams=MultiHeadClassifier.HParams(max_epochs_per_task=1)
    )
    results = mnist_setting.apply(method)
    assert results.to_log_dict()

    results: ClassIncrementalSetting.Results
    # There should be an improvement over the Method in `classifier.py`:
    assert 0.80 <= results.average_online_performance.objective <= 1.00
    assert 0.20 <= results.average_final_performance.objective <= 0.50


@slow
@pytest.mark.timeout(600)
def test_SL_track(sl_track_setting: SettingProxy[ClassIncrementalSetting]):
    """ Applies this Method to the Setting of the sl track of the competition.
    """
    method = ExampleTaskInferenceMethod(
        hparams=MultiHeadClassifier.HParams(max_epochs_per_task=1)
    )
    results = sl_track_setting.apply(method)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    assert 0.30 <= results.average_online_performance.objective <= 0.50
    assert 0.02 <= results.average_final_performance.objective <= 0.05
