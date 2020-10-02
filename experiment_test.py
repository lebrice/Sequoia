


import shlex
import sys
from typing import Optional, Type

import pytest

from common.config import Config
from methods import BaselineMethod, Method, RandomBaselineMethod, all_methods
from settings import Results, Setting, all_settings

from .experiment import Experiment


@pytest.mark.xfail(
    reason="@lebrice: I changed my mind on this. For example, it could make "
    "sense to have multiple methods called 'baseline' when a new Setting needs "
    "to create a new subclass of the BaselineMethod or a new Method altogether."
)
def test_no_collisions_in_method_names():
    assert len(set(method.get_name() for method in all_methods)) == len(all_methods)


def test_no_collisions_in_setting_names():
    assert len(set(setting.get_name() for setting in all_settings)) == len(all_settings)


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
    return


@pytest.fixture(params=all_methods)
def method_type(request, monkeypatch, set_argv_for_debug):
    method_class: Type[Method] = request.param
    # monkeypatch.setattr(method_class, "apply_to", mock_apply_to)
    return method_class


@pytest.fixture(params=all_settings)
def setting_type(request, monkeypatch, set_argv_for_debug):
    setting_class: Type[Setting] = request.param
    monkeypatch.setattr(setting_class, "apply", mock_apply)
    for method_type in setting_class.get_all_applicable_methods():
        pass
        # monkeypatch.setattr(method_type, "apply_to", mock_apply_to)
    return setting_class


@pytest.mark.parametrize("use_method_name", [False, True])
@pytest.mark.parametrize("use_setting_name", [False, True])
def test_combination_of_string_or_type(method_type: Optional[Type[Method]],
                                       use_method_name: bool,
                                       setting_type: Optional[Type[Setting]],
                                       use_setting_name: bool):
    
    method = method_type.get_name() if use_method_name else method_type
    setting = setting_type.get_name() if use_setting_name else setting_type
    
    experiment = Experiment(method=method, setting=setting)
    all_results = experiment.launch("--debug --fast_dev_run --batch-size 1")
    assert all_results == (method_type, setting_type)



@pytest.mark.parametrize("use_method_name", [False, True])
def test_none_setting(method_type: Optional[Type[Method]],
                      use_method_name: bool):
    method = method_type.get_name() if use_method_name else method_type
    experiment = Experiment(method=method, setting=None)
    all_results = experiment.launch("--debug --fast_dev_run --batch-size 1")
    for setting_type in method_type.get_all_applicable_settings():
        result = all_results[setting_type]
        assert result == (method_type, setting_type)


@pytest.mark.parametrize("use_setting_name", [False, True])
def test_none_method(setting_type: Optional[Type[Setting]],
                     use_setting_name: bool):
    setting = setting_type.get_name() if use_setting_name else setting_type
    experiment = Experiment(method=None, setting=setting)
    all_results = experiment.launch("--debug --fast_dev_run --batch-size 1")
    for method_type in setting_type.get_all_applicable_methods():
        result = all_results[method_type]
        assert result == (method_type, setting_type)

    # assert all_results == {
    #     method_type: (method_type, setting_type)
    #     for method_type in setting_type.get_all_applicable_methods()
    # }
