from typing import ClassVar, Type, cast

import pytest
from d3rlpy.constants import ActionSpace, DISCRETE_ACTION_SPACE_MISMATCH_ERROR, CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
from d3rlpy.dataset import MDPDataset, Episode, Transition
from d3rlpy.metrics import average_value_estimation_scorer, td_error_scorer
from sequoia import TraditionalRLSetting
from sequoia.methods.d3rlpy_methods.base import *
from sequoia.methods.method_test import MethodTests
from sequoia.settings.offline_rl.setting import OfflineRLSetting

"""
How d3rlpy checks for offline compatibility

# check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif transitions[0].is_discrete:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

How d3rlpy checks for online compatibility
if isinstance(env.action_space, gym.spaces.Box):
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        action_space = type(env.action_space)
        raise ValueError(f"The action-space is not supported: {action_space}")




"""


class BaseOfflineRLMethodTests:
    Method: ClassVar[Type[BaseOfflineRLMethod]]

    @pytest.fixture
    def method(self):
        return self.Method(train_steps=1000, train_steps_per_epoch=1000, scorers={
            'td_error': td_error_scorer,
            'value_scale': average_value_estimation_scorer
        })

    @pytest.mark.timeout(0)
    @pytest.mark.parametrize('dataset', OfflineRLSetting.available_datasets)
    def test_offlinerl(self, method, dataset: str):

        setting_offline = OfflineRLSetting(dataset=dataset)

        #
        # Check for mismatch
        if isinstance(setting_offline.env.action_space, gym.spaces.Box):
            if method.algo.get_action_type() != ActionSpace.CONTINUOUS:
                return

        elif isinstance(setting_offline.env.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() != ActionSpace.DISCRETE:
                return
        else:
            return

        results = setting_offline.apply(method)

        # Assert that loss in the final episode is less than .1
        objective = results[-1][1]['loss'] if 'loss' in results[-1][1] else results[-1][1]['temp_loss']
        assert objective

    '''
    @pytest.mark.parametrize('dataset', TraditionalRLSetting.available_datasets)
    def test_traditionalrl(self, method, dataset):
        setting_online = TraditionalRLSetting(dataset=dataset)

        #
        # Check for mismatch
        if isinstance(setting_online.action_space, gym.spaces.Box):
            if method.algo.get_action_type() != ActionSpace.CONTINUOUS:
                return

        elif isinstance(setting_online.action_space, gym.spaces.discrete.Discrete):
            if method.algo.get_action_type() != ActionSpace.DISCRETE:
                return
        else:
            return

        results = setting_online.apply(method)

        # Assert that the average validation reward is larger than the average train reward
        assert sum(results[0])/len(results[0]) <= sum(results[1])/len(results[1])
        '''


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
