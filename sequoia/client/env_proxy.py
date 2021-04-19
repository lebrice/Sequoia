"""TODO: Create an 'environment proxy' that relays observations / actions etc from a remote environment via gRPC.

For now this simply holds the 'remote' environment in memory.
"""
import itertools
from typing import Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from torch import Tensor

from sequoia.common.metrics import Metrics
from sequoia.settings import (
    Actions,
    ActionType,
    Environment,
    Observations,
    ObservationType,
    Results,
    Rewards,
    RewardType,
    Setting,
)


class EnvironmentProxy(Environment[ObservationType, ActionType, RewardType]):
    def __init__(self, env_fn, setting_type: Type[Setting]):
        # TODO: Actually interact with a given environment of the remote Setting
        # TODO: env_fn is just a callable that returns the actual env now, but the idea
        # is that it would perhaps be a handle/address/whatever which we could contact?
        self.__environment = env_fn()
        # TODO: Remove this if possible
        self._environment_type = type(self.__environment)
        self._setting_type = setting_type

        self.observation_space = self.get_attribute("observation_space")
        self.action_space = self.get_attribute("action_space")
        self.reward_space = self.get_attribute("reward_space")
        # TODO: Double check this also works for RL
        self.batch_size: Optional[int] = self.get_attribute("batch_size")

    def get_attribute(self, name: str):
        # TODO: actually get the value from the 'remote' env.
        return getattr(self.__environment, name)

    def reset(self) -> ObservationType:
        obs = self.__environment.reset()
        return obs

    def __len__(self) -> int:
        return len(self.__environment)

    def step(
        self, actions: ActionType
    ) -> Tuple[
        ObservationType,
        RewardType,
        Union[bool, Sequence[bool]],
        Union[Dict, Sequence[Dict]],
    ]:
        # Simulate converting things to a pickleable object?
        if isinstance(actions, Actions):
            actions = actions.numpy()
        actions_pkl = actions
        # TODO: Use some kind of gRPC endpoint.
        observations_pkl, rewards_pkl, done_pkl, info_pkl = self.__environment.step(
            actions_pkl
        )
        observations = self._setting_type.Observations(**observations_pkl)
        rewards = self._setting_type.Rewards(**rewards_pkl)
        done = np.array(done_pkl)
        info = np.array(info_pkl)
        return observations, rewards, done, info

    def __iter__(self):
        self.__environment.reset()
        return iter(self.__environment)
        # env_iterator = self._environment.__iter__()
        # print(f"Env iterator: {env_iterator}")
        # for episode_step in itertools.count():
        #     batch = next(env_iterator, None)

        #     if batch is None:
        #         self._environment.reset()
        #         break

        #     yield batch

    def send(self, actions: ActionType):
        if isinstance(actions, (Actions, Tensor)):
            actions = actions.numpy()
        actions_pkl = actions
        rewards_pkl = self.__environment.send(actions_pkl)
        rewards = self._setting_type.Rewards(**rewards_pkl)
        return rewards

    def close(self):
        self.__environment.close()

    @property
    def is_closed(self) -> bool:
        return self.get_attribute("is_closed")

    def render(self, *args, **kwargs):
        return self.__environment.render(*args, **kwargs)

    def get_results(self) -> Results:
        return self.__environment.get_results()

    def get_online_performance(self) -> List[Metrics]:
        return self.__environment.get_online_performance()

    def get_average_online_performance(self) -> Metrics:
        return self.__environment.get_average_online_performance()
