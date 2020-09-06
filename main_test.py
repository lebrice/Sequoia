


import pytest

from methods import BaselineMethod, Method, RandomBaselineMethod, all_methods
from settings import Results, Setting, all_settings

from .main import Experiment

from typing import Type
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

@pytest.fixture(params=all_methods)
def method_type(request, monkeypatch):
    method_class: Type[Method] = request.param
    monkeypatch.setattr(method_class, "apply_to", mock_apply_to)
    return method_class

@pytest.fixture(params=all_settings)
def setting_type(request, monkeypatch):
    setting_class: Type[Setting] = request.param
    monkeypatch.setattr(setting_class, "apply", mock_apply)
    return setting_class

def test_constructor_method_type_no_setting_type(method_type):
    experiment = Experiment(method=method_type, setting=None)
    all_results = experiment.launch()
    

def test_using_constructor_with_method_and_setting_type(method_type: Type[Method], setting_type: Type[Setting]):
    # method: Method = method_type.from_args("--debug --fast_dev_run")
    experiment = Experiment(method=method_type, setting=setting_type)
    all_results = experiment.launch()
    assert all_results == (method_type, setting_type)

def test_using_constructor_with_setting_no_method(setting_type: Type[Setting])
    experiment = Experiment(method=method_type, setting=None)
    all_results = experiment.launch()

def test_using_strings_for_both(method_type: Type[Method], setting_type: Type[Setting]):
    method: Method = method_type.from_args("--debug --fast_dev_run")
    experiment = Experiment(method=method_type, setting=None)
    all_results = experiment.launch()
    assert False, all_results
