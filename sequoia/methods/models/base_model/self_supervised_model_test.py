from typing import Dict, List, Tuple, Type

import pytest

from sequoia.conftest import id_fn, parametrize, slow
from sequoia.methods.aux_tasks import AE, EWC, VAE
from sequoia.methods.base_method import BaseMethod
from sequoia.settings.base import Results, Setting
from sequoia.settings.sl import TaskIncrementalSLSetting, TraditionalSLSetting
from sequoia.settings.sl.incremental import ClassIncrementalSetting

Method = BaseMethod
# Use 'Method' as an alias for the actual Method subclass under test. (since at
# the moment quite a few tests share some code.
# List of datasets that are currently supported for this method.
supported_datasets: List[str] = [
    "mnist",
    "fashion_mnist",
    "cifar10",
    "cifar100",
    "kmnist",
]


def test_get_applicable_settings():
    settings = Method.get_applicable_settings()
    assert ClassIncrementalSetting in settings
    assert TaskIncrementalSLSetting in settings
    assert TraditionalSLSetting in settings


@pytest.fixture(
    scope="module",
    params=[
        {},
        {VAE: 1},
        {AE: 1},
        {EWC: 1},
    ],  # no aux task.
    ids=id_fn,
)
def method_and_coefficients(request, tmp_path_factory):
    """Fixture that creates a method to be reused for the tests below as well
    as return the coefficients for each auxiliary task.
    """
    # Reuse the Method accross all tests below
    log_dir = tmp_path_factory.mktemp("log_dir")

    aux_task_coefficients = request.param

    args = f"""
    --debug
    --log_dir_root {log_dir}
    --default_root_dir {log_dir}
    --knn_samples 0
    --seed 123
    --fast_dev_run
    """
    for aux_task_name, coef in aux_task_coefficients.items():
        args += f"--{aux_task_name}.coef {coef} "

    return Method.from_args(args, strict=False), aux_task_coefficients


# @parametrize("dataset", get_dataset_params(Method, supported_datasets))


from sequoia.methods.method_test import key_fn


@slow
@parametrize("setting_type", sorted(Method.get_applicable_settings(), key=key_fn))
def test_fast_dev_run(
    method_and_coefficients: Tuple[Method, Dict[str, float]],
    setting_type: Type[Setting],
    test_dataset: str,
):
    """Performs a quick run with only one batch of train / val / test data and
    check that the 'Results' objects are ok.
    """
    method, aux_task_coefficients = method_and_coefficients
    if test_dataset not in setting_type.available_datasets:
        pytest.skip(msg=f"dataset {test_dataset} isn't available for this setting.")
    # Instantiate the setting
    setting: Setting = setting_type(dataset=test_dataset, nb_tasks=2)
    results: Results = setting.apply(method)
    validate_results(results, aux_task_coefficients)


def validate_results(results: Results, aux_task_coefficients: Dict[str, float]):
    """Makes sure that the results make sense for the method being tested.

    Checks that the Loss object has losses for each 'enabled' auxiliary task.

    Args:
        results (Results): A given Results object.
    """
    assert results is not None
    assert results.hparams is not None
    assert results.test_loss is not None

    for loss in results.task_losses:
        for aux_task_name, coef in aux_task_coefficients.items():
            assert aux_task_name in loss.losses
            aux_task_loss = loss.losses[aux_task_name]
            assert aux_task_loss.loss >= 0.0
            assert aux_task_loss._coefficient == coef
