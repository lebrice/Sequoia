""" WIP: Tests for the EWC Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, List, Type

import pytest
from torch.nn import Module
from avalanche.models import SimpleCNN, SimpleMLP

from sequoia.common import Config
from sequoia.conftest import xfail_param
from sequoia.settings.sl import IncrementalSLSetting, TaskIncrementalSLSetting

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .ewc import EWCMethod
from .patched_models import MTSimpleCNN, MTSimpleMLP

class TestEWCMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = EWCMethod
    ignored_parameter_differences: ClassVar[
        List[str]
    ] = _TestAvalancheMethod.ignored_parameter_differences + [
        "decay_factor",
    ]

    @classmethod
    @pytest.fixture(
        params=[
            SimpleCNN,
            SimpleMLP,
            xfail_param(
                MTSimpleCNN,
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
            xfail_param(
                MTSimpleMLP,
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
        ]
    )
    def method(cls, config: Config, request) -> AvalancheMethod:
        """Fixture that returns the Method instance to use when testing/debugging."""
        model_type = request.param
        return cls.Method(model=model_type, train_mb_size=10, train_epochs=1)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            SimpleCNN,
            SimpleMLP,
            # MTSimpleCNN,
            xfail_param(
                MTSimpleCNN,
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
            # MTSimpleMLP,
            xfail_param(
                MTSimpleMLP,
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
        ],
    )
    def test_short_task_incremental_setting(
        self,
        model_type: Type[Module],
        short_task_incremental_setting: TaskIncrementalSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
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
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
            # MTSimpleMLP,
            xfail_param(
                MTSimpleMLP,
                reason=(
                    "Shape Mismatch between the saved parameter importance and the "
                    "current weight tensor in EWC plugin."
                ),
            ),
        ],
    )
    def test_short_class_incremental_setting(
        self,
        model_type: Type[Module],
        short_class_incremental_setting: IncrementalSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
        results = short_class_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective

    # @pytest.mark.timeout(60)
    # @pytest.mark.parametrize(
    #     "model_type",
    #     [
    #         SimpleCNN,
    #         SimpleMLP,
    #         xfail_param(
    #             MTSimpleCNN,
    #             reason=(
    #                 "Shape Mismatch between the saved parameter importance and the "
    #                 "current weight tensor in EWC plugin."
    #             ),
    #         ),
    #         # MTSimpleMLP,
    #         xfail_param(
    #             MTSimpleMLP,
    #             reason=(
    #                 "Shape Mismatch between the saved parameter importance and the "
    #                 "current weight tensor in EWC plugin."
    #             ),
    #         ),
    #     ],
    # )
    # def test_short_continual_sl_setting(
    #     self,
    #     model_type: Type[Module],
    #     short_continual_sl_setting: ContinualSLSetting,
    #     config: Config,
    # ):
    #     super().test_short_continual_sl_setting(
    #         model_type=model_type,
    #         short_continual_sl_setting=short_continual_sl_setting,
    #         config=config,
    #     )

    # @pytest.mark.timeout(60)
    # @pytest.mark.parametrize(
    #     "model_type",
    #     [
    #         SimpleCNN,
    #         SimpleMLP,
    #         xfail_param(
    #             MTSimpleCNN,
    #             reason=(
    #                 "Shape Mismatch between the saved parameter importance and the "
    #                 "current weight tensor in EWC plugin."
    #             ),
    #         ),
    #         # MTSimpleMLP,
    #         xfail_param(
    #             MTSimpleMLP,
    #             reason=(
    #                 "Shape Mismatch between the saved parameter importance and the "
    #                 "current weight tensor in EWC plugin."
    #             ),
    #         ),
    #     ],
    # )
    # def test_short_discrete_task_agnostic_sl_setting(
    #     self,
    #     model_type: Type[Module],
    #     short_discrete_task_agnostic_sl_setting: DiscreteTaskAgnosticSLSetting,
    #     config: Config,
    # ):
    #     super().test_short_discrete_task_agnostic_sl_setting(
    #         model_type=model_type,
    #         short_discrete_task_agnostic_sl_setting=short_discrete_task_agnostic_sl_setting,
    #         config=config,
    #     )
