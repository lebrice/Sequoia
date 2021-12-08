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
# from typed_gym import Env, Space, VectorEnv

Observation = TypeVar("Observation")
Observation_co = TypeVar("Observation_co", covariant=True)
Action = TypeVar("Action")
Reward = TypeVar("Reward")
Reward_co = TypeVar("Reward_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

from sequoia.utils.generic_functions import stack
# from generic_functions import detach, stack, get_slice



@dataclass
class Transition(Generic[Observation, Action, Reward]):
    observation: Observation
    action: Action
    next_observation: Observation
    reward: Reward
    is_episode_end: bool = False


@dataclass
class Episode(Sequence[Transition[Observation, Action, Reward]]):
    observations: MutableSequence[Observation] = field(default_factory=list)
    actions: MutableSequence[Action] = field(default_factory=list)
    rewards: MutableSequence[Reward] = field(default_factory=list)
    infos: MutableSequence[dict] = field(default_factory=list)
    last_observation: Optional[Observation] = None

    # def append(self, transition: Transition) -> None:
    #     self.observations.append(transition.observation)
    #     self.actions.append(transition.action)
    #     self.rewards.append(transition.reward)
    #     if transition.is_episode_end:
    #         self.last_observation

    @overload
    def __getitem__(self, index: int) -> Transition[Observation, Action, Reward]:
        ...

    @overload
    def __getitem__(
        self, index: slice
    ) -> List[Transition[Observation, Action, Reward]]:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[
        Transition[Observation, Action, Reward],
        List[Transition[Observation, Action, Reward]],
    ]:
        if isinstance(index, int):
            if index not in range(0, len(self.observations)):
                raise IndexError(index)

            if index == len(self) - 1:
                if self.last_observation is None:
                    raise IndexError(index)
                next_observation = self.last_observation
            else:
                next_observation = self.observations[index + 1]

            return Transition(
                observation=self.observations[index],
                action=self.actions[index],
                next_observation=next_observation,
                reward=self.rewards[index],
                is_episode_end=index == len(self) - 1,
            )
        elif isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        raise IndexError

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
        )
