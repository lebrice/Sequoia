import random
import shlex
import sys
from pathlib import Path
from typing import Optional, Type

import pytest
from sequoia.common.config import Config
from sequoia.methods import Method, all_methods
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings import Results, Setting, all_settings
from sequoia.utils.serialization import Serializable

from .hpo_sweep import HPOSweep


class MockResults(Results):
    def __init__(self, hparams):
        self.haprams = hparams
        self._objective = random.random()

    @property
    def objective(self) -> float:
        return self._objective

    def make_plots(self):
        return {}

    def to_log_dict(self, verbose: bool = False):
        return {"hparams": self.hparams.to_dict() if isinstance(self.hparams, Serializable) else self.hparams, "objective": self.objective}

    def summary(self):
        return str(self.to_log_dict())


def mock_apply(self: Setting, method: Method, config: Config = None) -> Results:
    # 1. Configure the method to work on the setting.
    # method.configure(self)
    # 2. Train the method on the setting.
    # method.train(self)
    # 3. Evaluate the method on the setting and return the results.
    # return self.evaluate(method)
    # assert False, method.hparams
    return MockResults(getattr(method, "hparams", {}))
    # return type(method), type(self)


@pytest.fixture()
def set_argv_for_debug(monkeypatch):
    monkeypatch.setattr(sys, "argv", shlex.split("main.py --debug --fast_dev_run"))


@pytest.fixture(params=sorted(all_methods, key=str))
def method_type(request, monkeypatch, set_argv_for_debug):
    method_class: Type[Method] = request.param
    return method_class


from sequoia.methods.method_test import key_fn
@pytest.fixture(params=sorted(all_settings, key=key_fn))
def setting_type(request, monkeypatch, set_argv_for_debug):
    setting_class: Type[Setting] = request.param
    monkeypatch.setattr(setting_class, "apply", mock_apply)
    # TODO: Not sure what this was doing, but I think it was important that all methods
    # get imported here.
    for method_type in setting_class.get_applicable_methods():
        pass
    return setting_class


@pytest.mark.skip(reason="BUG: seems to make other tests hang, because of Orion's bug.")
def test_launch_sweep_with_constructor(
    method_type: Optional[Type[Method]],
    setting_type: Optional[Type[Setting]],
    tmp_path: Path,
):
    if not method_type.is_applicable(setting_type):
        pytest.skip(
            msg=f"Skipping test since Method {method_type} isn't applicable on settings of type {setting_type}."
        )

    if issubclass(method_type, RandomBaselineMethod):
        pytest.skip("BUG: RandomBaselineMethod has a hparam space that causes the HPO algo to go into an infinite loop.")
        return

    experiment = HPOSweep(
        method=method_type,
        setting=setting_type,
        database_path=tmp_path / "debug.pkl",
        config=Config(debug=True),
        max_runs=3,
    )
    best_hparams, best_performance = experiment.launch(["--debug"])
    assert best_hparams
    assert best_performance
