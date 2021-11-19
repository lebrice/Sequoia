from typing import Type, ClassVar, List, Tuple, Dict, Union, Optional

import gym
from d3rlpy.algos import *
import numpy as np
from d3rlpy.metrics import td_error_scorer, average_value_estimation_scorer
from gym import Space
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from sequoia import Method, TraditionalRLSetting, Environment, Observations, Actions, Rewards
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

    def __init__(self, train_steps: int = 1_000_000,
                 train_steps_per_epoch=1_000_000,
                 scorers: dict = None,
                 use_gpu: bool = False):
        super().__init__()
        self.train_steps = train_steps
        self.train_steps_per_epoch = train_steps_per_epoch
        self.scorers = scorers
        self.algo = type(self).Algo(use_gpu=use_gpu)

    def configure(self, setting: OfflineRLSetting)-> None:
        super().configure(setting)
        self.setting = setting

    def fit(self, train_env, valid_env) -> List[Tuple[int, Dict[str, float]]]:
        if isinstance(self.setting, OfflineRLSetting):
            return self.algo.fit(train_env,
                                 eval_episodes=valid_env,
                                 n_steps=self.train_steps,
                                 n_steps_per_epoch=self.train_steps_per_epoch,
                                 scorers=self.scorers)
        else:
            train_env, valid_env = RecordEpisodeStatistics(OfflineRLWrapper(train_env)), \
                                   RecordEpisodeStatistics(OfflineRLWrapper(valid_env))
            self.algo.fit_online(env=train_env, eval_env=valid_env, n_steps=self.train_steps)

    def get_actions(self, obs: np.ndarray, action_space: Space) -> np.ndarray:
        # TODO: bug with TraditionalRLTests here.... what is this supposed to return?
        return self.algo.predict(obs)

    # TODO: Remove this when get_actions is fixed
    def test(self, test_env: Environment[Observations, Actions, Optional[Rewards]]):
        pass

    # TODO: save() method?

"""
D3RLPY Methods: work on OfflineRL and TraditionalRL assumptions
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

# Quick example using DQN for offline cart-pole


def online_example():
    setting_online = TraditionalRLSetting(dataset="Cartpole-v0")
    method = DQNMethod(train_steps=1, train_steps_per_epoch=1, scorers={
        'td_error': td_error_scorer,
        'value_scale': average_value_estimation_scorer
    })

    results = setting_online.apply(method)
    print(results)


if __name__ == "__main__":
    online_example()
