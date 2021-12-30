from __future__ import annotations

from typing import Iterator, List, Optional, TypeVar

import gym
from sequoia.common.spaces.utils import batch_space

# TODO: There's a difference between a buffer with Episode as the item, and a buffer with Transition as the item!
from sequoia.common.typed_gym import _Action, _Env, _Observation_co, _Reward
from torch.utils.data import DataLoader, IterableDataset

from .episode import Episode, Transition
from .episode_collector import EpisodeCollector
from .policy import Policy, RandomPolicy
from .replay_buffer import ReplayBuffer

T = TypeVar("T")


def make_env_loader(
    env: _Env[_Observation_co, _Action, _Reward],
    policy: Policy[_Observation_co, _Action],
    batch_size: int,
    buffer_size: int = 10_000,
    max_episodes: int = None,
    max_steps: int = None,
    seed: int = None,
) -> OffPolicyTransitionsLoader[_Observation_co, _Action, _Reward]:
    dataset = OffPolicyTransitionsDataset(
        env=env,
        policy=policy,
        max_steps=max_steps,
        max_episodes=max_episodes,
        buffer_size=buffer_size,
        seed=seed,
    )
    loader = OffPolicyTransitionsLoader(dataset=dataset, batch_size=batch_size)
    return loader


class OffPolicyTransitionsDataset(
    gym.Wrapper, IterableDataset[Transition[_Observation_co, _Action, _Reward]]
):
    def __init__(
        self,
        env: _Env[_Observation_co, _Action, _Reward],
        policy: Policy[_Observation_co, _Action],
        batch_size: int = 1,  # NOTE: Number of transitions to yield at each step in the iteration.
        buffer_size: int = 10_000,
        max_steps: int = None,
        max_episodes: int = None,
        seed: int = None,
    ) -> None:
        super().__init__(env=env)
        self.env = env
        self.policy = policy

        item_space = Transition.space_for_env(env)
        item_space.seed(seed)
        if seed is not None:
            # Seed the item space, since it is used to populate the replay buffer.
            item_space.seed(seed)

        self.dataset: ReplayBuffer[Transition[_Observation_co, _Action, _Reward]] = ReplayBuffer(
            item_space=item_space, capacity=buffer_size, seed=seed
        )
        self.episode_generator = EpisodeCollector(
            self.env,
            policy=policy or RandomPolicy(),
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
        if policy is not None:
            self.send(policy)

        if seed is not None:
            self.seed(seed)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Transition[_Observation_co, _Action, _Reward]]:
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

            new_policy: Optional[Policy[_Observation_co, _Action]]
            new_policy = yield sampled_transitions

            if new_policy is not None:
                self.episode_generator.send(new_policy)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        self.policy = new_policy
        self.episode_generator.send(new_policy)


class OffPolicyEpisodeDataset(
    gym.Wrapper, IterableDataset[Episode[_Observation_co, _Action, _Reward]]
):
    """ TODO: No need to create this class ourselves. Might be better to just use the one from
    d3rlpy.
    
    Also, ReplayBuffer doesn't quite work atm with an item space where items have different lengths.
    """
    def __init__(
        self,
        env: _Env[_Observation_co, _Action, _Reward],
        policy: Policy[_Observation_co, _Action],
        batch_size: int = 1,  # NOTE: Number of episodes to yield at each 'step'.
        buffer_size: int = 10_000,
        max_steps: int = None,
        max_episodes: int = None,
        seed: int = None,
    ) -> None:
        super().__init__(env=env)
        self.env = env
        self.policy = policy

        item_space = Transition.space_for_env(env)
        item_space.seed(seed)
        if seed is not None:
            # Seed the item space, since it is used to populate the replay buffer.
            item_space.seed(seed)

        # TODO: This doesn't work quite yet: The 'item space' here is for transitions, which doesn't
        # match with the 'Episode' objects.
        self.dataset: ReplayBuffer[Episode[_Observation_co, _Action, _Reward]] = ReplayBuffer(
            item_space=item_space, capacity=buffer_size, seed=seed
        )
        self.episode_generator = EpisodeCollector(
            self.env,
            policy=policy or RandomPolicy(),
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
        if policy is not None:
            self.send(policy)

        if seed is not None:
            self.seed(seed)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Episode[_Observation_co, _Action, _Reward]]:
        # Populate the replay buffer with transitions, until we're able to produce a full batch.
        for i, new_episode in enumerate(self.episode_generator):
            # NOTE: Difference from the above: Here we add the episode as a sample in the buffer.
            self.dataset.add_reservoir([new_episode])

            sampled_episodes = self.dataset.sample(n_samples=self.batch_size)
            assert all(isinstance(sampled_episode, Episode) for sampled_episode in sampled_episodes)

            new_policy: Optional[Policy[_Observation_co, _Action]]
            new_policy = yield sampled_episodes

            if new_policy is not None:
                self.episode_generator.send(new_policy)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        self.policy = new_policy
        self.episode_generator.send(new_policy)


class OffPolicyTransitionsLoader(DataLoader[Transition[_Observation_co, _Action, _Reward]]):
    def __init__(
        self,
        dataset: OffPolicyTransitionsDataset[_Observation_co, _Action, _Reward],
        *,
        batch_size: int,
        policy: Policy[_Observation_co, _Action] = None,
        max_steps: int = None,
        max_episodes: int = None,
        seed: Optional[int] = None,
        **kwargs,
    ):

        self.batch_size = batch_size
        self._seed = seed
        self._max_steps = max_steps
        self._max_episodes = max_episodes
        # NOTE: Some things can't be changed here.
        kwargs.update(num_workers=0, collate_fn=None)
        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)
        assert self.dataset is dataset

        if policy is not None:
            self.send(policy)

        if self._seed is not None:
            self.seed(seed)

        self.item_space = batch_space(Transition.space_for_env(dataset.env), n=self.batch_size)

    def seed(self, seed: Optional[int]) -> List[int]:
        self.dataset.seed(seed)

    def __iter__(self) -> Iterator[Transition[_Observation_co, _Action, _Reward]]:
        # Populate the replay buffer with transitions, until we're able to produce a full batch.
        # TODO: Need to batch the samples from the dataset.
        return iter(self.dataset)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        self.episode_generator.send(new_policy)


class OffPolicyEpisodeLoader(DataLoader[Episode[_Observation_co, _Action, _Reward]]):
    def __init__(
        self,
        dataset: OffPolicyTransitionsDataset[_Observation_co, _Action, _Reward],
        *,
        batch_size: int,
        policy: Policy[_Observation_co, _Action] = None,
        max_steps: int = None,
        max_episodes: int = None,
        seed: Optional[int] = None,
        **kwargs,
    ):

        self.batch_size = batch_size
        self._seed = seed
        self._max_steps = max_steps
        self._max_episodes = max_episodes
        # NOTE: Some things can't be changed here.
        kwargs.update(num_workers=0, collate_fn=None)
        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)
        assert self.dataset is dataset

        if policy is not None:
            self.send(policy)

        if self._seed is not None:
            self.seed(seed)

        self.item_space = batch_space(Transition.space_for_env(dataset.env), n=self.batch_size)

    def seed(self, seed: Optional[int]) -> List[int]:
        self.dataset.seed(seed)

    def __iter__(self) -> Iterator[Transition[_Observation_co, _Action, _Reward]]:
        # Populate the replay buffer with transitions, until we're able to produce a full batch.
        # TODO: Need to batch the samples from the dataset.
        return iter(self.dataset)

    def send(self, new_policy: Policy[Observation, _Action]) -> None:
        self.episode_generator.send(new_policy)
