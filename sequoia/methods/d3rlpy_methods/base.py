from typing import Type, ClassVar, List, Tuple, Dict, Optional, Union

import gym
from d3rlpy.algos import *
import numpy as np
from d3rlpy.dataset import MDPDataset
from gym import Space
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from sequoia import Method, Environment, Observations, Actions, Rewards, TraditionalRLSetting
from sequoia.settings.offline_rl.setting import OfflineRLSetting


class OfflineRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.x

    def reset(self):
        observation = super().reset()
        return observation.x

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation.x, reward.y, done, info


class BaseOfflineRLMethod(Method, target_setting=OfflineRLSetting):
    Algo: ClassVar[Type[AlgoBase]] = AlgoBase

    def __init__(self,
                 train_steps: int = 1_000_000,
                 train_steps_per_epoch=1_000_000,
                 test_steps=1_000,
                 scorers: dict = None,
                 use_gpu: bool = False):
        super().__init__()
        self.train_steps = train_steps
        self.train_steps_per_epoch = train_steps_per_epoch
        self.test_steps = test_steps
        self.scorers = scorers
        self.offline_metrics = None

        # TODO: add support for all algo params, not just common ones
        self.algo = type(self).Algo(use_gpu=use_gpu)

    def configure(self, setting: OfflineRLSetting) -> None:
        super().configure(setting)
        self.setting = setting

    def fit(self,
            train_env: Union[Environment[Observations, Actions, Rewards], MDPDataset],
            valid_env: Union[Environment[Observations, Actions, Rewards], MDPDataset]) -> None:
        """
        Fit self.algo on training and evaluation environment
        Works for both sequoia environments and d3rlpy datasets
        """
        if isinstance(self.setting, OfflineRLSetting):
            self.offline_metrics = self.algo.fit(train_env,
                                 eval_episodes=valid_env,
                                 n_steps=self.train_steps,
                                 n_steps_per_epoch=self.train_steps_per_epoch,
                                 scorers=self.scorers)
        else:
            train_env, valid_env = RecordEpisodeStatistics(OfflineRLWrapper(train_env)), \
                                   RecordEpisodeStatistics(OfflineRLWrapper(valid_env))
            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        """
        Return actions predicted by self.algo for given observation and action space
        """
        obs = np.expand_dims(obs, axis=0)
        action = np.asarray(self.algo.predict(obs)).squeeze(axis=0)
        return action

    def test(self, test_env):
        """
            Test self.algo on given test_env for self.test_steps iterations
        """
        test_env = RecordEpisodeStatistics(OfflineRLWrapper(test_env))

        obs = test_env.reset()
        for _ in range(self.test_steps):
            obs, reward, done, info = test_env.step(self.get_actions(obs, action_space=test_env.action_space))
            if done:
                break
        test_env.close()

"""
D3RLPY Methods: target OfflineRL and TraditionalRL assumptions
"""


class DQNMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DQN


class DoubleDQNMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DoubleDQN


class DDPGMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DDPG


class TD3Method(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = TD3


class SACMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = SAC


class DiscreteSACMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteSAC


class CQLMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = CQL


class DiscreteCQLMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteCQL


class BEARMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BEAR


class AWRMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = AWR


class DiscreteAWRMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteAWR


class BCMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BC


class DiscreteBCMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteBC


class BCQMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = BCQ


class DiscreteBCQMethod(BaseOfflineRLMethod):
    Algo: ClassVar[Type[AlgoBase]] = DiscreteBCQ


if __name__ == '__main__':
    setting_online = TraditionalRLSetting(dataset='CartPole-v0')
    alg = DQNMethod(train_steps=5, train_steps_per_epoch=5, test_steps=5)
    results = setting_online.apply(alg)
    print(results.objective)

