from typing import (
    Generic,
    List,
    Optional,
    TypeVar,
    MutableSequence,
    Union,
    Dict,
    overload,
)
from dataclasses import dataclass, field
from typing import Sequence

from torch.utils import data

# from typed_gym import Env, Space, VectorEnv

Observation = TypeVar("Observation")
Observation_co = TypeVar("Observation_co", covariant=True)
Action = TypeVar("Action")
Reward = TypeVar("Reward")
Reward_co = TypeVar("Reward_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

from sequoia.utils.generic_functions import stack
from sequoia.common.batch import Batch

from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import _Env, _Space

from gym import spaces
from gym.vector import VectorEnv
import numpy as np
from typing import Tuple


def get_reward_space(env: _Env[Observation, Action, Reward]) -> _Space[Reward]:
    if hasattr(env, "reward_space") and env.reward_space is not None:
        return env.reward_space
    reward_range: Tuple[float, float] = getattr(env, "reward_range", (-np.inf, np.inf))
    num_envs = env.num_envs if isinstance(env.unwrapped, VectorEnv) else None
    return spaces.Box(
        reward_range[0],
        reward_range[1],
        dtype=float,
        shape=(num_envs) if num_envs is not None else (),
    )


def get_num_envs(env: _Env) -> Optional[int]:
    if isinstance(env.unwrapped, VectorEnv):
        return env.num_envs
    else:
        return None


@dataclass(frozen=True)
class Transition(Batch, Generic[Observation_co, Action, Reward]):
    observation: Observation_co
    action: Action
    reward: Reward
    next_observation: Observation_co
    info: Optional[Dict] = None
    done: bool = False

    @classmethod
    def space_for_env(cls, env: _Env[Observation_co, Action, Reward]) -> _Space["Transition[Observation_co, Action, Reward]"]:
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


# NOTE: Making `Episode` a Sequence[Transition[Observation,Action,Reward]] causes a bug in PL atm:
# forward receives a Transition, rather than an Observation.


@dataclass(frozen=True)
class Episode(
    Sequence[Transition[Observation, Action, Reward]],
    Generic[Observation, Action, Reward],
):
    observations: MutableSequence[Observation] = field(default_factory=list)
    actions: MutableSequence[Action] = field(default_factory=list)
    rewards: MutableSequence[Reward] = field(default_factory=list)
    infos: MutableSequence[dict] = field(default_factory=list)
    last_observation: Optional[Observation] = None

    model_versions: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Length of the episode, as-in number of transitions."""
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1
        # return n_obs

    def stack(self) -> "Episode[Observation, Action, Reward]":
        return Episode(
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
        Transition[Observation, Action, Reward],
        Sequence[Transition[Observation, Action, Reward]],
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


from sequoia.utils.generic_functions import set_slice, get_slice
from dataclasses import fields


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
class StackedEpisode(Generic[Observation, Action, Reward]):
    observations: Observation
    actions: Action
    rewards: Reward
    infos: Sequence[dict]
    last_observation: Observation
    model_versions: Sequence[int]

    def __len__(self) -> int:
        """Length of the episode, as-in number of transitions."""
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1
