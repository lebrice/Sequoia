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
        if isinstance(setting_offline.env.action_space, gym.spaces.Box):
            if method.algo.get_action_type() not in  {ActionSpace.CONTINUOUS, ActionSpace.BOTH}:
                pytest.skip("This setting requires continuous action space algorithm")

        elif isinstance(setting_offline.env.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() not in {ActionSpace.DISCRETE, ActionSpace.BOTH}:
                pytest.skip("This setting requires discrete action space algorithm")
        else:
            pytest.skip("Invalid setting action space")

        results = setting_offline.apply(method)

        # Difficult to set a meaningful threshold for 1 step fit
        assert isinstance(results.objective, float)

    @pytest.mark.parametrize('dataset', TraditionalRLSetting.available_datasets)
    def test_traditionalrl(self, method, dataset):

        # BC is a strictly offline method
        if isinstance(method, (BCMethod, BCQMethod, DiscreteBCMethod, DiscreteBCQMethod)):
            pytest.skip("This method only works on OfflineRLSetting")

        setting_online = TraditionalRLSetting(dataset=dataset, test_max_steps=10)

        #
        # Check for mismatch
        if isinstance(setting_online.action_space, gym.spaces.Box):
            if method.algo.get_action_type() != ActionSpace.CONTINUOUS:
                pytest.skip("This setting requires continuous action space algorithm")

        elif isinstance(setting_online.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() != ActionSpace.DISCRETE:
                pytest.skip("This setting requires discrete action space algorithm")
        else:
            pytest.skip("Invalid setting action space")

        results = setting_online.apply(method)

        # Difficult to set a meaningful threshold for 1 step fit
        assert isinstance(results.objective, float) or isinstance(results.objective, int)


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
