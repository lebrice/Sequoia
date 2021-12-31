from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Dict,
    Generic,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from gym.spaces.space import Space

import numpy as np
from gym import spaces
from gym.vector import VectorEnv
from torch import Tensor
from sequoia.common.batch import Batch
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.spaces.utils import get_batch_type_for_item_type, get_item_type_for_batch_type
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

from sequoia.utils.utils import flatten_dict

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
                "observation": getattr(
                    env, "single_observation_space", env.observation_space
                ),
                "action": getattr(env, "single_action_space", env.action_space),
                "reward": getattr(env, "single_reward_space", reward_space),
                "next_observation": env.observation_space,
                "info": spaces.Dict(),
                "done": spaces.Box(
                    False, True, dtype=bool, shape=()
                ),  # note: single bool.
            },
            dtype=cls,
        )


# NOTE: Making `Episode` a Sequence[Transition[_Observation,_Action,_Reward]] causes a bug in PL atm:
# forward receives a Transition, rather than an _Observation.


@dataclass
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

    @property
    def length(self) -> int:
        return len(self.observations)

    def __len__(self) -> int:
        """Length of the episode, as-in number of transitions."""
        return self.length

    def stack(self) -> "StackedEpisode[_Observation, _Action, _Reward]":
        return StackedEpisode(
            observations=stack(self.observations),
            actions=stack(self.actions),
            rewards=stack(self.rewards),
            infos=np.array(
                self.infos
            ),  # TODO: List of empty dicts in case infos is empty.
            last_observation=self.last_observation,
            model_versions=np.array(self.model_versions),
        )

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[
        Transition[_Observation, _Action, _Reward],
        Sequence[Transition[_Observation, _Action, _Reward]],
    ]:  
        # NOTE: Slice indexing isn't actually supported atm.
        if not isinstance(index, (int, np.integer)):
            raise NotImplementedError(index, type(index))
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

@get_slice.register(Episode)
def _get_episode_slice(value: Episode[_Observation_co, _Action, _Reward], indices: Sequence[int]) -> Episode[_Observation_co, _Action, _Reward]:
    # TODO: Not sure if it's a good idea to return the last observation in the episode as well..
    
    if isinstance(indices, slice):
        indices = np.arange(len(value))[indices]
    indices = np.array(indices)

    ep_len = len(value)
    if (ep_len-1) in indices:
        last_observation = value.last_observation
    else:
        last_observation = None

    return type(value)(
        observations=[value.observations[i] for i in indices],
        actions=[value.actions[i] for i in indices],
        rewards=[value.rewards[i] for i in indices],
        infos=[value.infos[i] for i in indices],
        last_observation=last_observation,
        model_versions=[value.model_versions[i] for i in indices],
    )


@set_slice.register(Episode)
def _set_episode_slice(target: Episode[_Observation_co, _Action, _Reward], indices: Sequence[int], values: Episode[_Observation_co, _Action, _Reward]) -> Episode[_Observation_co, _Action, _Reward]:
    assert len(indices) == len(values), (indices, len(values), values)

    for dest_index, source_index in zip(indices, range(len(values))):
        target.observations[dest_index] = values.observations[source_index]
        target.actions[dest_index] = values.actions[source_index]
        target.rewards[dest_index] = values.rewards[source_index]
        target.infos[dest_index] = values.infos[source_index]
        target.model_versions[dest_index] = values.model_versions[source_index]

    if len(target) - 1 in indices:
        target.last_observation = values.last_observation


@set_slice.register(Transition)
def _set_transition_slice(
    target: Transition, indices: Sequence[int], values: Transition
):
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


from collections import Counter


@dataclass(frozen=True)
class StackedEpisode(Batch, Generic[_Observation, _Action, _Reward]):
    observations: _Observation
    actions: _Action
    rewards: _Reward
    infos: Sequence[dict]
    last_observation: _Observation
    model_versions: Sequence[int]

    # TODO: __len__ here is a bit ambiguous, between the number of fields (Batch) and number of
    # observations (Episode)
    def __len__(self) -> int:
        raise NotImplementedError(
            f"Disabling using `len` on this StackedEpisode class, since it can be ambiguous."
            f"Use `length` for number of episodes, and len(v.keys()) for number of fields."
        )

    @cached_property
    def length(self) -> int:
        """Length of the episode, as-in number of transitions."""
        n_model_versions = len(self.model_versions)
        self.batch_size
        flattened_shapes_dict = flatten_dict(self.shapes, separator=".")
        batch_sizes = {
            k: shape[0]
            for k, shape in flattened_shapes_dict.items()
            if shape is not None and len(shape) >= 1
        }
        batch_size_counter = Counter(batch_sizes.values())
        most_common_batch_size, count = batch_size_counter.most_common(1)[0]
        # just to double-check:
        assert most_common_batch_size == len(self.model_versions), (self.shapes, self.model_versions)
        n_infos = len(self.infos)
        assert most_common_batch_size == n_infos, (most_common_batch_size, n_infos)
        return most_common_batch_size

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[
        Transition[_Observation, _Action, _Reward],
        Sequence[Transition[_Observation, _Action, _Reward]],
    ]:
        if not isinstance(index, int):
            return super().__getitem__(index)

        obs = get_slice(self.observations, index)
        action = get_slice(self.actions, index)
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



""" IDEA: What about creating a kind of immutable object that holds stacked observations,
actions, rewards, last_observations, etc, in a more efficient way than a simple List[Episode]?

For example, say we create empty arrays for 1000 obs/actions/rewards, and then each "episode"
indexes inside these arrays?

NOTE: This sounds cool, but isn't really needed atm.
"""