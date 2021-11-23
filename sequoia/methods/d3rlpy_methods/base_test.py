import pytest
from d3rlpy.constants import ActionSpace

from sequoia.methods.d3rlpy_methods.base import *
from sequoia.settings.offline_rl.setting import OfflineRLSetting
from sequoia import TraditionalRLSetting


class BaseOfflineRLMethodTests:
    Method: ClassVar[Type[BaseOfflineRLMethod]]

    @pytest.fixture
    def method(self):
        return self.Method(train_steps=1, train_steps_per_epoch=1)

    @pytest.mark.parametrize('dataset', OfflineRLSetting.available_datasets)
    def test_offlinerl(self, method, dataset: str):

        setting_offline = OfflineRLSetting(dataset=dataset)

        #
        # Check for mismatch
        if isinstance(setting_offline.action_space, gym.spaces.Box):
            if method.algo.get_action_type() != ActionSpace.CONTINUOUS:
                pytest.skip("This setting requires continuous action space algorithm")

        elif isinstance(setting_offline.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() != ActionSpace.DISCRETE:
                pytest.skip("This setting requires discrete action space algorithm")
        else:
            return

        results = setting_offline.apply(method)

        # Check that the metric dict for our 1 step epoch is not None

        epoch_metrics = results[-1][1]
        assert epoch_metrics is not None
        assert isinstance(epoch_metrics, dict)

    @pytest.mark.parametrize('dataset', TraditionalRLSetting.available_datasets)
    def test_traditionalrl(self, method, dataset):

        # BC is a strictly offline method
        if type(method) in {BCMethod, BCQMethod, DiscreteBCMethod, DiscreteBCQMethod}:
            pytest.skip("This method only works on OfflineRLSetting")

        setting_online = TraditionalRLSetting(dataset=dataset)

        #
        # Check for mismatch
        if isinstance(setting_online.action_space, gym.spaces.Box):
            if method.algo.get_action_type() != ActionSpace.CONTINUOUS:
                pytest.skip("This setting requires continuous action space algorithm")

        elif isinstance(setting_online.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() != ActionSpace.DISCRETE:
                pytest.skip("This setting requires discrete action space algorithm")
        else:
            return

        results = setting_online.apply(method)

        assert results


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
