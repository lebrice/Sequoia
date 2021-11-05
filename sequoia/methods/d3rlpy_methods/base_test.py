from typing import ClassVar, Type

import pytest
from d3rlpy.metrics import average_value_estimation_scorer, td_error_scorer
from sequoia import TraditionalRLSetting
from sequoia.methods.d3rlpy_methods.base import BaseOfflineRLMethod, DQNMethod
from sequoia.methods.method_test import MethodTests
from sequoia.settings.offline_rl.setting import OfflineRLSetting


class BaseOfflineRLMethodTests:
    Method: ClassVar[Type[BaseOfflineRLMethod]]

    @pytest.fixture
    def method(self):
       return  self.Method(scorers={
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