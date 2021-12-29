from dataclasses import dataclass, field
from typing import Dict, Generic, List, MutableSequence, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from gym import spaces
from gym.vector import VectorEnv
from sequoia.common.batch import Batch
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import (
    _Action,
    _Env,
    _Observation,
    _Observation_co,
    _Reward,
    _Reward_co,
    _Space,
)
from sequoia.utils.generic_functions import stack
from torch.utils import data

from .utils import get_num_envs, get_reward_space

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


@dataclass(frozen=True)
class Transition(Batch, Generic[_Observation_co, _Action, _Reward]):
    observation: _Observation_co
    action: _Action
    reward: _Reward
    next_observation: _Observation_co
    info: Optional[Dict] = None
    done: bool = False

    @classmethod
    def space_for_env(
        cls, env: _Env[_Observation_co, _Action, _Reward]
    ) -> _Space["Transition[_Observation_co, _Action, _Reward]"]:
        # num_envs = get_num_envs(env)
        reward_space = get_reward_space(env)
        return TypedDictSpace(
            spaces={
                "observation": getattr(env, "single_observation_space", env.observation_space),
                "action": getattr(env, "single_action_space", env.action_space),
                "reward": getattr(env, "single_reward_space", reward_space),
                "next_observation": env.observation_space,
                "info": spaces.Dict(),
                "done": spaces.Box(False, True, dtype=bool, shape=()),  # note: single bool.
            },
            dtype=cls,
        )


# NOTE: Making `Episode` a Sequence[Transition[_Observation,_Action,_Reward]] causes a bug in PL atm:
# forward receives a Transition, rather than an _Observation.


@dataclass(frozen=True)
class Episode(
    Sequence[Transition[_Observation, _Action, _Reward]],
    Generic[_Observation, _Action, _Reward],
):
    observations: MutableSequence[_Observation] = field(default_factory=list)
    actions: MutableSequence[_Action] = field(default_factory=list)
    rewards: MutableSequence[_Reward] = field(default_factory=list)
    infos: MutableSequence[dict] = field(default_factory=list)
    last_observation: Optional[_Observation] = None

    model_versions: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Length of the episode, as-in number of transitions."""
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1
        # return n_obs

    def stack(self) -> "Episode[_Observation, _Action, _Reward]":
        return StackedEpisode(
            observations=stack(self.observations),
            actions=stack(self.actions),
            rewards=stack(self.rewards),
            infos=stack(self.infos),  # TODO: List of empty dicts in case infos is empty.
            last_observation=self.last_observation,
            model_versions=stack(self.model_versions),
        )

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[
        Transition[_Observation, _Action, _Reward],
        Sequence[Transition[_Observation, _Action, _Reward]],
    ]:
        if not isinstance(index, int):
            raise NotImplementedError(index)
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} is out of range! (len is {len(self)})")

        obs = self.observations[index]
        action = self.actions[index]
        next_obs = (
            self.last_observation
            if self.last_observation is not None and index == len(self.observations) - 1
            else self.observations[index + 1]
        )
        reward = self.rewards[index]
        info = self.infos[index] if self.infos else {}
        done = index == (len(self.observations) - 1)
        return Transition(
            observation=obs,
            action=action,
            next_observation=next_obs,
            reward=reward,
            info=info,
            done=done,
        )


from dataclasses import fields

from sequoia.utils.generic_functions import get_slice, set_slice


@set_slice.register(Transition)
def _set_transition_slice(target: Transition, indices: Sequence[int], values: Transition):
    for f in fields(target):
        k = f.name
        target_v = getattr(target, k)
        values_v = getattr(values, k)
        set_slice(target_v, indices=indices, values=values_v)


@get_slice.register(Transition)
def _get_transition_slice(target: Transition, indices: Sequence[int]) -> Transition:
    result = {}
    for f in fields(target):
        k = f.name
        target_v = getattr(target, k)
        sliced_v = get_slice(target_v, indices=indices)
        result[k] = sliced_v
    return type(target)(**result)


@dataclass(frozen=True)
class StackedEpisode(Generic[_Observation, _Action, _Reward]):
    observations: _Observation
    actions: _Action
    rewards: _Reward
    infos: Sequence[dict]
    last_observation: _Observation
    model_versions: Sequence[int]

    def __len__(self) -> int:
        """Length of the episode, as-in number of transitions."""
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1
