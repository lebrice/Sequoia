""" Idea: a simple wrapper that counts the number of steps and episodes, with
optional arguments used to limit the number of steps until the env is done.
"""

from typing import (Dict, Generator, Generic, Iterable, Optional, Sequence,
                    Tuple, Type, TypeVar, Union)

import gym
from torch.utils.data import IterableDataset

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

# TODO: @lebrice Create a wrapper that stores the last state in the `info` dict.
# depending on the `done` value as well.


class EnvDataset(gym.Wrapper, IterableDataset, Generic[ObservationType, ActionType, RewardType]):
    """ Wrapper that exposes a Gym environment as an IterableDataset.

    This makes it possible to iterate over a gym env with an Active DataLoader.
    """

    def __init__(self,
                 env: gym.Env,
                 max_episodes: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 ):
        super().__init__(env=env)
        # Maximum number of episodes to perform in the environment.
        self.max_episodes = max_episodes
        # Maximum number of steps to perform in the environment.
        self.max_steps = max_steps
        # Number of steps performed in the environment.
        self.n_steps: int = 0
        # Number of times the `send` method was called, i.e. number of actions
        # taken in the environment.
        self.n_actions: int = 0
        # Number of episodes performed in the environment.
        # Starts at -1 so that an initial reset brings it to 0.
        self.n_episodes: int = -1

        self._observation: Optional[ObservationType] = None 
        self._action: Optional[ActionType] = None
        self._reward: Optional[RewardType] = None
        self._done: Optional[Union[bool, Sequence[bool]]] = None
        self._info: Optional[Union[Dict, Sequence[Dict]]] = None

    def step(self, action) -> Tuple[ObservationType,
                                    RewardType,
                                    Union[bool, Sequence[bool]],
                                    Union[Dict, Sequence[Dict]]]:
        self._action = action
        self._observation, self._reward, self._done, self._info = super().step(self._action)
        self.n_steps += 1
        assert self._observation is not None
        assert self._reward is not None
        assert self._done is not None
        assert self._info is not None
        return self._observation, self._reward, self._done, self._info

    def __next__(self) -> Tuple:
        return self.step(self.action)

    def __iter__(self) -> Iterable[Tuple[ObservationType,
                                         Union[bool, Sequence[bool]],
                                         Union[Dict, Sequence[Dict]]]]:
        # Reset the env if it hasn't been called before iterating.
        if self.n_episodes == -1:
            self.reset()

        while not self.reached_episode_limit or self.reached_step_limit:
            # Perform an episode.
            while not (self._done or self.reached_step_limit):
                # TODO: @lebrice Isn't there something fishy going on here? I'm
                # not sure that we're giving back the right reward for the right
                # action and observation?

                # TODO: This should be a 'push' model, steps should occur when
                # the action is received, the corresponding reward be
                # immediately returned, and the next yield statement in the
                # iterator should give back the rest ? (Need to figure this out)
                if self._observation is None or self._done is None:
                    raise RuntimeError(
                        "You need to send an action using the `send` method "
                        "every time you get a value from the dataset! "
                        "Otherwise, you can also pass in a policy to use when "
                        "an action isn't given."
                    )
                assert self._observation is not None
                assert self._done is not None
                assert self._info is not None

                action = yield (self._observation, self._done, self._info)
                
                assert action is None, (
                    "Send actions to the env using the `send` method on the "
                    "env, not on the iterator itself!"
                )
                # IDEA: 'Delete' these attributes, to force users to send an
                # action using either `send` or `step` between each iteration.
                self._observation = None
                self._done = None
                self._info = None

            self.episode_count += 1
            # NOTE: It's important that we call `self.env.reset` rather than
            # `self.reset` because that would reset the number of episodes and
            # steps performed and mess up the checks above.
            self.env.reset()

        self.close()

    def send(self, action: ActionType) -> RewardType:
        assert action is not None, "Don't send a None action!"
        self.n_actions += 1
        self.step(action)
        assert self._reward is not None
        return self._reward

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps is not None:
            return self.n_steps >= self.max_steps
        return False
    
    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes is not None:
            return self.n_episodes >= self.max_episodes
        return False

    def reset(self, **kwargs) -> ObservationType:
        self._observation = super().reset(**kwargs)
        self._reward = None
        self._done = False
        self._info = {}
        self.n_episodes += 1
        return self._observation

    def close(self) -> None:
        # This will stop the iterator on the next step.
        self.max_steps = 0
        super().close()
