""" WIP: Tests for the AR1 Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

import pytest
from torch.nn import Module

from sequoia.common.config import Config
from sequoia.conftest import xfail_param
from sequoia.settings.sl import TaskIncrementalSLSetting

from .ar1 import AR1Method
from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .patched_models import MTSimpleCNN, MTSimpleMLP, SimpleCNN, SimpleMLP


@pytest.mark.xfail(reason="AR1 isn't super well supported yet.")
class TestAR1Method(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = AR1Method

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            xfail_param(
                SimpleCNN,
                reason="seems like the model in AR1 is supposed to be larger?",
            ),
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
        short_task_incremental_setting: TaskIncrementalSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type)
        results = short_task_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective
