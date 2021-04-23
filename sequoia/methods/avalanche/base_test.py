import inspect
from inspect import Signature, _empty, getsourcefile
from typing import ClassVar, Type

import pytest
from torch.nn import Module

from avalanche.models import MTSimpleCNN, MTSimpleMLP, SimpleCNN, SimpleMLP
from avalanche.training.strategies import BaseStrategy

from sequoia.common.config import Config
from sequoia.conftest import xfail_param
from sequoia.settings.passive import ClassIncrementalSetting, TaskIncrementalSetting
from .base import AvalancheMethod


class TestAvalancheMethod:
    Method: ClassVar[Type[AvalancheMethod]] = AvalancheMethod

    def test_hparams_have_same_defaults_as_in_avalanche(self):
        strategy_type: Type[BaseStrategy] = self.Method.strategy_class
        method = self.Method()
        strategy_constructor: Signature = inspect.signature(strategy_type.__init__)
        strategy_init_params = strategy_constructor.parameters
        for parameter_name, parameter in strategy_init_params.items():
            if parameter.default is _empty:
                continue
            assert hasattr(method, parameter_name)
            method_value = getattr(method, parameter_name)
            # Ignore mismatches in some parameters, like `device`.
            if parameter_name in ["device", "eval_mb_size", "criterion"]:
                continue

            assert method_value == parameter.default, (
                f"{self.Method.__name__} in Sequoia has different default value for "
                f"hyper-parameter '{parameter_name}' than in Avalanche: \n"
                f"\t{method_value} != {parameter.default}\n"
                f"Path to sequoia implementation: {getsourcefile(self.Method)}\n"
                f"Path to SB3 implementation: {getsourcefile(strategy_type)}\n"
            )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            SimpleCNN,
            SimpleMLP,
            xfail_param(
                MTSimpleCNN,
                reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            ),
            xfail_param(
                MTSimpleMLP,
                reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            ),
        ],
    )
    def test_short_task_incremental_setting(
        self,
        model_type: Type[Module],
        short_task_incremental_setting: TaskIncrementalSetting,
        config: Config,
    ):
        method = self.Method(model=model_type)
        results = short_task_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            SimpleCNN,
            SimpleMLP,
            xfail_param(
                MTSimpleCNN,
                reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            ),
            xfail_param(
                MTSimpleMLP,
                reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            ),
        ],
    )
    def test_short_class_incremental_setting(
        self,
        model_type: Type[Module],
        short_class_incremental_setting: ClassIncrementalSetting,
        config: Config,
    ):
        method = self.Method(model=model_type)
        results = short_class_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective
