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
        return self.Method(scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    @pytest.mark.parametrize('dataset', OfflineRLSetting.available_datasets)
    def test_offlinerl(self, method, dataset: str):
        # TODO check compatibility of method and dataset

        setting_offline = OfflineRLSetting(dataset=dataset)
        results = setting_offline.apply(method)

        assert results.objective > 100

    @pytest.mark.parametrize('dataset', TraditionalRLSetting.available_datasets)
    def test_traditionalrl(self, method, dataset):
        setting_online = TraditionalRLSetting(dataset=dataset)
        results = setting_online.apply(method)
        # TODO: Validate results
        assert results.objective > 0


class test_dqn(BaseOfflineRLMethodTests):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DQNMethod


class test_DoubleDQNMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DoubleDQNMethod


class test_DDPGMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DDPGMethod


class test_TD3Method(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = TD3Method


class test_SACMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = SACMethod


class test_DiscreteSACMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteSACMethod


class CQLMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = CQLMethod


class test_DiscreteCQLMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteCQLMethod


class test_BEAR(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BEARMethod


class test_AWRMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = AWRMethod


class test_DiscreteAWRMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteAWRMethod


class test_BCMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCMethod


class test_DiscreteBCMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCMethod


class test_BCQMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = BCQMethod


class test_DiscreteBCQMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteBCQMethod


class test_RandomPolicyMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = RandomPolicyMethod


class test_DiscreteRandomPolicyMethod(BaseOfflineRLMethod):
    Method: ClassVar[Type[BaseOfflineRLMethod]] = DiscreteRandomPolicyMethod


