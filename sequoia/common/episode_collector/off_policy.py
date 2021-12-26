from functools import singledispatch
from typing import Iterator, List, Optional, Tuple, TypeVar

import numpy as np
from gym import spaces
from gym.vector import VectorEnv
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.common.typed_gym import _Env, _Space, _VectorEnv
from torch.utils.data import DataLoader

from .episode import Action, Observation, Reward, Transition
from .episode_collector import EpisodeCollector
from .policy import Policy, RandomPolicy
from .replay_buffer import ReplayBuffer

T = TypeVar("T")


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

from gym.vector.utils import batch_space

# TODO: There's a difference between a buffer with Episode as the item, and a buffer with Transition as the item!


class OffPolicyTransitionsLoader(DataLoader[Transition[Observation, Action, Reward]]):
    def __init__(
        self,
        env: _Env[Observation, Action, Reward],
        *,
        batch_size: int,
        buffer_size: int = 10_000,
        policy: Policy[Observation, Action] = None,
        max_steps: int = None,
        max_episodes: int = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self._env = env
        self.batch_size = batch_size
        self._seed = seed
        self._capacity = buffer_size
        self._max_steps = max_steps
        self._max_episodes = max_episodes

        # NOTE: Use the single space? or batched space?
        num_envs = get_num_envs(env)
        self.reward_space = get_reward_space(env)
        item_space = TypedDictSpace(
            spaces={
                "observation": getattr(
                    env, "single_observation_space", env.observation_space
                ),
                "action": getattr(env, "single_action_space", env.action_space),
                "reward": getattr(env, "single_reward_space", self.reward_space),
                "next_observation": env.observation_space,
                "info": spaces.Dict(),
                "done": spaces.Box(False, True, dtype=bool, shape=()),
            },
            dtype=Transition,
        )
        if seed is not None:
            # Seed the item space, since it is used to populate the replay buffer.
            item_space.seed(seed)

        dataset = ReplayBuffer(
            item_space=item_space, capacity=buffer_size, seed=seed
        )
        self.episode_generator = EpisodeCollector(
            self._env,
            policy=policy or RandomPolicy(),
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
        # NOTE: Some things can't be changed here.
        kwargs.update(num_workers=0, collate_fn=None)
        super().__init__(
            dataset=dataset, batch_size=batch_size, **kwargs
        )
        self.dataset: ReplayBuffer[Transition[Observation, Action, Reward]]
        assert self.dataset is dataset

        if policy is not None:
            self.send(policy)

        if self._seed is not None:
            self.seed(seed)

        # The space of the batches that this will yield.
        self.item_space = batch_space(item_space, n=batch_size)

    def seed(self, seed: Optional[int]) -> List[int]:
        self._seed = seed
        self._env.seed(seed)
        self._env.action_space.seed(seed)
        self._env.observation_space.seed(seed)
        self.reward_space.seed(seed)
        self.dataset.seed(seed)

    def __iter__(self) -> Iterator[Transition[Observation, Action, Reward]]:
        # Populate the replay buffer with transitions, until we're able to produce a full batch.
        for i, new_episode in enumerate(self.episode_generator):
            # transitions = list(new_episode)
            # NOTE: Episode := List[Transition]
            self.dataset.add_reservoir(new_episode)

            if len(self.dataset) < self.batch_size:
                # Buffer doesn't contain enough data yet, skip to the next iteration.
                continue

            sampled_transitions = self.dataset.sample(n_samples=self.batch_size)
            assert isinstance(sampled_transitions, Transition), sampled_transitions

            new_policy: Optional[Policy[Observation, Action]]
            new_policy = yield sampled_transitions

            if new_policy is not None:
                self.episode_generator.send(new_policy)

    def send(self, new_policy: Policy[Observation, Action]) -> None:
        self.episode_generator.send(new_policy)
