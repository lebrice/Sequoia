from typing import (
    Generic,
    List,
    Optional,
    TypeVar,
    MutableSequence,
    Union,
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


@dataclass
class Episode(Generic[Observation, Action, Reward]):
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
            infos=stack(self.infos),
            last_observation=self.last_observation,
            model_versions=stack(self.model_versions)
        )


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
