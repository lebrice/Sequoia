import shlex
import sys
from pathlib import Path
from typing import Optional, Type

import pytest


from sequoia.conftest import slow
from sequoia.common.config import Config
from sequoia.methods import Method, all_methods
from sequoia.methods.baseline_method import BaselineMethod
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings import Results, Setting, all_settings

from .experiment import Experiment, get_method_names
method_names = get_method_names()


@pytest.mark.xfail(
    reason="@lebrice: I changed my mind on this. For example, it could make "
    "sense to have multiple methods called 'baseline' when a new Setting needs "
    "to create a new subclass of the BaselineMethod or a new Method altogether."
)
def test_no_collisions_in_method_names():
    assert len(set(method.get_name() for method in all_methods)) == len(all_methods)


def test_no_collisions_in_setting_names():
    assert len(set(setting.get_name() for setting in all_settings)) == len(all_settings)


def test_applicable_methods():
    from sequoia.methods import BaselineMethod
    from sequoia.settings import TraditionalSLSetting

    assert BaselineMethod in TraditionalSLSetting.get_applicable_methods()


def mock_apply(self: Setting, method: Method, config: Config) -> Results:
    # 1. Configure the method to work on the setting.
    # method.configure(self)
    # 2. Train the method on the setting.
    # method.train(self)
    # 3. Evaluate the method on the setting and return the results.
    # return self.evaluate(method)
    return type(method), type(self)


@pytest.fixture()
def set_argv_for_debug(monkeypatch):
    monkeypatch.setattr(sys, "argv", shlex.split("main.py --debug --fast_dev_run"))


@pytest.fixture(params=all_methods)
def method_type(request, monkeypatch, set_argv_for_debug):
    method_class: Type[Method] = request.param
    return method_class


@pytest.fixture(params=all_settings)
def setting_type(request, monkeypatch, set_argv_for_debug):
    setting_class: Type[Setting] = request.param
    monkeypatch.setattr(setting_class, "apply", mock_apply)
    for method_type in setting_class.get_applicable_methods():
        pass
    return setting_class


def test_experiment_from_args(
    method_type: Optional[Type[Method]], setting_type: Optional[Type[Setting]]
):
    """ Test that when parsing the 'Experiment' from the command-line, the
    `setting` and `method` fields get set to the classes corresponding to their
    names.
    """
    # method = method_type.get_name()
    method_name = [k for k, v in method_names.items() if v is method_type][0]
    setting = setting_type.get_name()
    if not method_type.is_applicable(setting_type):
        pytest.skip(
            msg=f"Skipping test since Method {method_type} isn't applicable on "
            f"settings of type {setting_type}."
        )
    experiment = Experiment.from_args(f"--setting {setting} --method {method_name}")
    assert experiment.method is method_type
    assert experiment.setting is setting_type


def test_launch_experiment_with_constructor(
    method_type: Optional[Type[Method]], setting_type: Optional[Type[Setting]]
):
    if not method_type.is_applicable(setting_type):
        pytest.skip(
            msg=f"Skipping test since Method {method_type} isn't applicable on "
            f"settings of type {setting_type}."
        )
    experiment = Experiment(method=method_type, setting=setting_type)
    all_results = experiment.launch("--debug --fast_dev_run --batch_size 1")
    assert all_results == (method_type, setting_type)


@slow
@pytest.mark.timeout(300)
def test_none_setting(method_type: Optional[Type[Method]], tmp_path: Path, monkeypatch):
    """ Test that leaving the Setting unset runs on all applicable setting. """
    method = method_type.get_name()

    for setting_type in method_type.get_applicable_settings():
        monkeypatch.setattr(setting_type, "apply", mock_apply)

    all_results = Experiment.main(
        f"--method {method} --debug --fast_dev_run " f"--log_dir {tmp_path}"
    )

    for setting_type in method_type.get_applicable_settings():
        monkeypatch.setattr(setting_type, "apply", mock_apply)
        result = all_results[(setting_type, method_type)]
        assert result == (method_type, setting_type)


@slow
@pytest.mark.timeout(300)
def test_none_method(setting_type: Optional[Type[Setting]]):
    """ Test that leaving the method unset runs all applicable methods on the
    setting.
    """
    setting = setting_type.get_name()
    all_results = Experiment.main(
        f"--setting {setting} --debug --fast_dev_run --batch-size 1"
    )
    for method_type in setting_type.get_applicable_methods():
        result = all_results[(setting_type, method_type)]
        assert result == (method_type, setting_type)

    # assert all_results == {
    #     method_type: (method_type, setting_type)
    #     for method_type in setting_type.get_applicable_methods()
    # }
