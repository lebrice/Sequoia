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

@dataclass(frozen=True)
class Transition(Batch, Generic[Observation_co, Action, Reward]):
    observation: Observation_co
    action: Action
    reward: Reward
    next_observation: Observation_co
    info: Optional[Dict] = None
    done: bool = False


# NOTE: Making `Episode` a Sequence[Transition[Observation,Action,Reward]] causes a bug in PL atm:
# forward receives a Transition, rather than an Observation.


@dataclass(frozen=True)
class Episode(Batch,
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
        """ Length of the episode, as-in number of transitions. """
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1

    def stack(self) -> "Episode[Observation, Action, Reward]":
        return Episode(
            observations=stack(self.observations),
            actions=stack(self.actions),
            rewards=stack(self.rewards),
            infos=stack(self.infos), # TODO: List of empty dicts in case infos is empty.
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
        return Transition(
            observation=self.observations[index],
            action=self.actions[index],
            next_observation=(
                self.last_observation if index == len(self) else self.observations[index+1]
            ),
            reward=self.rewards[index],
            info=self.infos[index] if self.infos else {},
            done=index == (len(self) - 1),
        )

from sequoia.utils.generic_functions import set_slice, get_slice
from dataclasses import fields


@set_slice.register(Transition)
def _(target: Transition, indices: Sequence[int], values: Transition):
    for f in fields(target):
        k = f.name
        target_v = getattr(target, k)
        values_v = getattr(values, k)
        set_slice(target_v, indices=indices, values=values_v)


@get_slice.register(Transition)
def _(target: Transition, indices: Sequence[int]) -> Transition:
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
        """ Length of the episode, as-in number of transitions. """
        n_obs = len(self.observations)
        return n_obs if self.last_observation is not None else n_obs - 1
