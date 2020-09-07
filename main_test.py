


import shlex
import sys
from typing import Type

import pytest

from methods import BaselineMethod, Method, RandomBaselineMethod, all_methods
from settings import Results, Setting, all_settings

from .main import Experiment


@pytest.mark.xfail(
    reason="@lebrice: I changed my mind on this. For example, it could make "
    "sense to have multiple methods called 'baseline' when a new Setting needs "
    "to create a new subclass of the BaselineMethod or a new Method altogether."
)
def test_no_collisions_in_method_names():
    assert len(set(method.get_name() for method in all_methods)) == len(all_methods)


def test_no_collisions_in_setting_names():
    assert len(set(setting.get_name() for setting in all_settings)) == len(all_settings)


def mock_apply_to(self: Method, setting: Setting) -> Results:
    """ Applies this method to the particular experimental setting.
    
    Extend this class and overwrite this method to create a different method.        
    """
    # 1. Configure the method to work on the setting.
    # self.configure(setting)
    # 2. Train the method on the setting.
    # self.train(setting)
    # 3. Evaluate the model on the setting and return the results.
    # return setting.evaluate(self)
    return type(self), type(setting)

def mock_apply(self: Setting, method: Method) -> Results:
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
    monkeypatch.setattr(method_class, "apply_to", mock_apply_to)
    return method_class


@pytest.fixture(params=all_settings)
def setting_type(request, monkeypatch, set_argv_for_debug):
    setting_class: Type[Setting] = request.param
    monkeypatch.setattr(setting_class, "apply", mock_apply)
    return setting_class


def test_method_and_setting_types(method_type: Type[Method], setting_type: Type[Setting], set_argv_for_debug):
    # using both: simplest.
    experiment = Experiment(method=method_type, setting=setting_type)
    all_results = experiment.launch()
    assert all_results == (method_type, setting_type)

    # Running all settings
    experiment = Experiment(method=method_type, setting=None)
    all_results = experiment.launch()
    assert all_results == {
        setting: (method_type, setting) for setting in method_type.get_all_applicable_settings()
    }
    # Running all methods
    experiment = Experiment(method=None, setting=setting_type)
    all_results = experiment.launch()
    assert all_results == {
        method_type: (method_type, setting_type) for method_type in setting_type.get_all_applicable_methods()
    }

    # Setting both to strings.
    experiment = Experiment(method=method_type.get_name(), setting=setting_type.get_name())
    all_results = experiment.launch()
    assert all_results == (method_type, setting_type)
    # Setting either to a string
    experiment = Experiment(method=method_type.get_name(), setting=setting_type)
    all_results = experiment.launch()
    assert all_results == (method_type, setting_type)
    experiment = Experiment(method=method_type, setting=setting_type.get_name())
    all_results = experiment.launch()
    assert all_results == (method_type, setting_type)

    # Running all settings (passing a string)
    experiment = Experiment(method=method_type.get_name(), setting=None)
    all_results = experiment.launch()
    assert all_results == {
        setting: (method_type, setting) for setting in method_type.get_all_applicable_settings()
    }

    # Running all methods
    experiment = Experiment(method=None, setting=setting_type.get_name())
    all_results = experiment.launch()
    assert all_results == {
        method_type: (method_type, setting_type) for method_type in setting_type.get_all_applicable_methods()
    }
