from functools import singledispatch
from numpy.random.mtrand import sample
from torch.utils.data import DataLoader
from typing import (
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)
from sequoia.common.spaces.typed_dict import TypedDictSpace

from sequoia.common.typed_gym import _Env, _Space, _VectorEnv
from .episode import Action, Observation, Reward, T, Transition
from .episode_collector import EpisodeCollector
from .replay_buffer import ReplayBuffer
from .policy import Policy, RandomPolicy

T = TypeVar("T")
from gym.vector import VectorEnv
import numpy as np
from gym import spaces


@singledispatch
def get_reward_space(env: _Env[Observation, Action, Reward]) -> _Space[Reward]:
    if hasattr(env, "reward_space") and env.reward_space is not None:
        return env.reward_space
    reward_range: Tuple[float, float] = getattr(env, "reward_range", (-np.inf, np.inf))
    num_envs = env.num_envs if isinstance(env.unwrapped, VectorEnv) else None
    return spaces.Box(reward_range[0], reward_range[1], dtype=float, shape=(num_envs) if num_envs is not None else ())


@get_reward_space.register(VectorEnv)
def _(env: _VectorEnv[Observation, Action, Reward]) -> _Space[Reward]:
    if hasattr(env, "reward_space") and env.reward_space is not None:
        return env.reward_space
    reward_range: Tuple[float, float] = getattr(env, "reward_range", (-np.int, np.inf))
    return spaces.Box(
        reward_range[0], reward_range[1], dtype=float, shape=(env.num_envs,)
    )


def get_num_envs(env: _Env) -> Optional[int]:
    if isinstance(env.unwrapped, VectorEnv):
        return env.num_envs
    else:
        return None

# TODO: There's a difference between a buffer with Episode as the item, and a buffer with Transition as the item!


class ExperienceReplayLoader(DataLoader[Transition[Observation, Action, Reward]]):
    def __init__(
        self,
        env: _Env[Observation, Action, Reward],
        batch_size: int,
        buffer_size: int = 10_000,
        max_steps: int = None,
        max_episodes: int = None,
        policy: Policy[Observation, Action] = None,
    ):
        self.env = env
        self.batch_size = batch_size
        self._capacity = buffer_size
        self._max_steps = max_steps
        self._max_episodes = max_episodes

        # NOTE: Use the single space? or batched space?
        num_envs = get_num_envs(env)
        item_space = TypedDictSpace(
            spaces={
                "observation": getattr(
                    env, "single_observation_space", env.observation_space
                ),
                "action": getattr(env, "single_action_space", env.action_space),
                "reward": getattr(env, "single_reward_space", get_reward_space(env)),
                "next_observation": env.observation_space,
                "info": spaces.Dict(),
                "done": spaces.Box(False, True, dtype=bool, shape=()),
            },
            dtype=Transition
        )
        dataset = ReplayBuffer(item_space=item_space, capacity=buffer_size)
        self.episode_generator = EpisodeCollector(
            self.env,
            policy=RandomPolicy(),
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=0, collate_fn=None)
        self.dataset: ReplayBuffer[Transition[Observation, Action, Reward]]
        assert self.dataset is dataset

        if policy is not None:
            self.send(policy)

    def __iter__(self) -> Iterator[Transition[Observation, Action, Reward]]:
        # Populate the replay buffer with transitions, until we're able to produce a full batch.
        for i, new_episode in enumerate(self.episode_generator):
            transitions = list(new_episode)
            # NOTE: Episode := List[Transition]
            self.dataset.add_reservoir(transitions)

            if len(self.dataset) < self.batch_size:
                # Buffer doesn't contain enough data yet, skip to the next iteration.
                continue

            sampled_transitions = self.dataset.sample(n_samples=self.batch_size)
            
            assert isinstance(sampled_transitions, Transition)
            assert isinstance(sampled_transitions.observation, np.ndarray)
            assert sampled_transitions.observation.shape == self.batch_size


            new_policy: Optional[Policy[Observation, Action]] = yield sampled_transitions

            if new_policy is not None:
                self.episode_generator.send(new_policy)

    def send(self, new_policy: Policy[Observation, Action]) -> None:
        self.episode_generator.send(new_policy)
