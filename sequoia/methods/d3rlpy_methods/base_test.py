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


class TestDoubleDQNMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DoubleDQNMethod


class TestDDPGMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DDPGMethod


class TestTD3Method(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = TD3Method


class TestSACMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = SACMethod


class TestDiscreteSACMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteSACMethod


class TestCQLMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = CQLMethod


class TestDiscreteCQLMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteCQLMethod


class TestBEARMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BEARMethod


class TestAWRMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = AWRMethod


class TestDiscreteAWRMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteAWRMethod


class TestBCMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCMethod


class TestDiscreteBCMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCMethod


class TestBCQMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCQMethod


class TestDiscreteBCQMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCQMethod


class TestRandomPolicyMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = RandomPolicyMethod


class TestDiscreteRandomPolicyMethod(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteRandomPolicyMethod


