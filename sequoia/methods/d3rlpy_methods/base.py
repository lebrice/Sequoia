from typing import ClassVar, Type, Union

import gym
import numpy as np

try:
    from d3rlpy.algos import *
    from d3rlpy.dataset import MDPDataset
except ImportError as err:
    raise RuntimeError(f"You need to have `d3rlpy` installed to use these methods.") from err

from gym import Space
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from sequoia import Actions, Environment, Method, Observations, Rewards
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

    def __init__(
        self,
        train_steps: int = 1_000_000,
        train_steps_per_epoch=1_000_000,
        test_steps=1_000,
        scorers: dict = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.train_steps = train_steps
        self.train_steps_per_epoch = train_steps_per_epoch
        self.test_steps = test_steps
        self.scorers = scorers
        self.offline_metrics = None
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        self.algo = None

    def configure(self, setting: OfflineRLSetting) -> None:
        super().configure(setting)
        self.setting = setting
        self.algo = type(self).Algo(use_gpu=self.use_gpu, **self.kwargs)

    def fit(
        self,
        train_env: Union[Environment[Observations, Actions, Rewards], MDPDataset],
        valid_env: Union[Environment[Observations, Actions, Rewards], MDPDataset],
    ) -> None:
        """
        Fit self.algo on training and evaluation environment
        Works for both gym environments and d3rlpy datasets
        """
        if isinstance(self.setting, OfflineRLSetting):
            self.offline_metrics = self.algo.fit(
                train_env,
                eval_episodes=valid_env,
                n_steps=self.train_steps,
                n_steps_per_epoch=self.train_steps_per_epoch,
                scorers=self.scorers,
            )
        else:
            train_env = RecordEpisodeStatistics(OfflineRLWrapper(train_env))
            valid_env = RecordEpisodeStatistics(OfflineRLWrapper(valid_env))
            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: Union[np.ndarray, Observations], action_space: Space) -> np.ndarray:
        """
        Return actions predicted by self.algo for given observation and action space
        """
        if isinstance(obs, Observations):
            obs = obs.x
        obs = np.expand_dims(obs, axis=0)
        action = np.asarray(self.algo.predict(obs)).squeeze(axis=0)
        return action


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
