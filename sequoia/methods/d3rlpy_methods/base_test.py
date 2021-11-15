from typing import ClassVar, Type

import pytest
from d3rlpy.metrics import average_value_estimation_scorer, td_error_scorer
from sequoia import TraditionalRLSetting
from sequoia.methods.d3rlpy_methods.base import *
from sequoia.methods.method_test import MethodTests
from sequoia.settings.offline_rl.setting import OfflineRLSetting


class BaseOfflineRLMethodTests:
    Method: ClassVar[Type[BaseOfflineRLMethod]]

    @pytest.fixture
    def method(self):
        return self.Method(train_steps=1000, train_steps_per_epoch=1000, scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    @pytest.mark.parametrize('dataset', OfflineRLSetting.available_datasets)
    def test_offlinerl(self, method, dataset: str):
        # TODO check compatibility of method and dataset

        setting_offline = OfflineRLSetting(dataset=dataset)
        results = setting_offline.apply(method)

        # assert results.objective > 0

    @pytest.mark.parametrize('dataset', TraditionalRLSetting.available_datasets)
    def test_traditionalrl(self, method, dataset):
        setting_online = TraditionalRLSetting(dataset=dataset)

        # TODO check compatibility

        results = setting_online.apply(method)
        # TODO: Validate results
        # assert results.objective > 0


class TestDQNMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DQNMethod


class TestDoubleDQNMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DoubleDQNMethod


class TestDDPGMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DDPGMethod


class TestTD3Method(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = TD3Method


class TestSACMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = SACMethod


class TestDiscreteSACMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteSACMethod


class TestCQLMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = CQLMethod


class TestDiscreteCQLMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteCQLMethod


class TestBEARMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BEARMethod


class TestAWRMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = AWRMethod


class TestDiscreteAWRMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteAWRMethod


class TestBCMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCMethod


class TestDiscreteBCMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCMethod


class TestBCQMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCQMethod


class TestDiscreteBCQMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCQMethod


class TestRandomPolicyMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = RandomPolicyMethod


class TestDiscreteRandomPolicyMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteRandomPolicyMethod


