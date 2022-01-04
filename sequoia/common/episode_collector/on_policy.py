from __future__ import annotations
from torch.utils.data import DataLoader, ChainDataset
from torch.utils.data.dataloader import (
    Dataset,
    T_co,
    Sampler,
    _collate_fn_t,
    _worker_init_fn_t,
)
from typing import List, Optional, Sequence, Union, Iterator, Optional

from torch.utils.data.dataset import IterableDataset
from sequoia.common.episode_collector.policy import Policy
from sequoia.common.episode_collector.update_strategy import detach_actions_strategy
from sequoia.common.typed_gym import (
    _Env,
    _VectorEnv,
    _Observation_co,
    _Action,
    _Reward,
    _Reward,
)
from .episode import Episode, StackedEpisode
from .episode_collector import (
    EpisodeCollector,
    PolicyUpdateStrategy,
    do_nothing_strategy,
    redo_forward_pass_strategy,
)
import gym


def make_env_loader(
    env: _Env[_Observation_co, _Action, _Reward],
    policy: Policy[_Observation_co, _Action],
    max_episodes_per_iteration: int = None,
    max_steps_per_iteration: int = None,
    batch_size: int = 1,
    seed: int = None,
    what_to_do_after_update: PolicyUpdateStrategy = detach_actions_strategy,
    empty_batch_interval: int = 0,
) -> OnPolicyEpisodeLoader[_Observation_co, _Action, _Reward]:
    """[summary]

    [extended_summary]

    Parameters
    ----------
    env : _Env[_Observation_co, _Action, _Reward]
        [description]
    policy : Policy[_Observation_co, _Action]
        [description]
    max_episodes_per_iteration : int, optional
        [description], by default None
    max_steps_per_iteration : int, optional
        [description], by default None
    batch_size : int, optional
        [description], by default 1
    seed : int, optional
        [description], by default None
    what_to_do_after_update : PolicyUpdateStrategy, optional
        [description], by default detach
    empty_batch_interval : int, optional
        Yield an empty batch at the given interval. Defaults to 0, in which case the loader doesn't
        yield empty batches. Setting this to a positive integer can be useful when there is a delay
        between the producer (dataloader) and the consumer (model updates), so that the episodes can
        stay fully on-policy even after an update. (It's basically a hack to make On-Policy training
        with PyTorch-Lightning easier to do.)

    Returns
    -------
    OnPolicyEpisodeLoader[_Observation_co, _Action, _Reward]
        [description]
    """
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    dataset = OnPolicyEpisodeDataset(
        env=env,
        policy=policy,
        max_iter_steps=max_steps_per_iteration,
        max_iter_episodes=max_episodes_per_iteration,
        what_to_do_after_update=what_to_do_after_update,
    )
    loader = OnPolicyEpisodeLoader(
        dataset=dataset, batch_size=batch_size, empty_batch_interval=empty_batch_interval
    )
    return loader


class OnPolicyEpisodeDataset(
    gym.Wrapper, IterableDataset[StackedEpisode[_Observation_co, _Action, _Reward]]
):
    def __init__(
        self,
        env: _Env[_Observation_co, _Action, _Reward],
        policy: Policy[_Observation_co, _Action],
        max_iter_steps: Optional[int] = None,
        max_iter_episodes: Optional[int] = None,
        what_to_do_after_update: PolicyUpdateStrategy = do_nothing_strategy,
    ) -> None:
        super().__init__(env=env)
        self.env = env
        self.max_iter_steps = max_iter_steps
        self.max_iter_episodes = max_iter_episodes
        self.policy = policy
        self.what_to_do_after_update = what_to_do_after_update
        self._episode_generator = self._create_generator()

    def _create_generator(self) -> EpisodeCollector[_Observation_co, _Action, _Reward]:
        return EpisodeCollector(
            self.env,
            policy=self.policy,
            max_steps=self.max_iter_steps,
            max_episodes=self.max_iter_episodes,
            what_to_do_after_update=self.what_to_do_after_update,
        )

    def __iter__(self) -> Iterator[StackedEpisode[_Observation_co, _Action, _Reward]]:
        if self._episode_generator is not None:
            self._episode_generator.close()
        # note: No access to the max_steps or max_episodes args of EpisodeCollector for now.
        # Could be confusing with the max steps or episodes when interacting with the env.
        self._episode_generator = self._create_generator()
        return self._episode_generator

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        self.policy = new_policy
        self._episode_generator.send(new_policy)

    def __add__(
        self,
        other: Union[
            _Env[_Observation_co, _Action, _Reward],
            "OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward]",
        ],
    ) -> "OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward]":
        if not isinstance(other, (OnPolicyEpisodeDataset, gym.Env)):
            raise NotImplementedError(other)
        # TODO: Maybe create a ChainDataset that has a Send method, which sends to the current
        # iterator.
        # TODO: Use one of the MultiEnv Wrappers?
        from sequoia.settings.rl.discrete.multienv_wrappers import (
            ConcatEnvsWrapper,
            RoundRobinWrapper,
            RandomMultiEnvWrapper,
        )

        raise NotImplementedError("TODO: Justify this use-case, even though we *could* add it.")
        return type(self)(env=ConcatEnvsWrapper(envs=[self.env, other]), policy=self.policy)


import itertools


class OnPolicyEpisodeLoader(DataLoader[StackedEpisode[_Observation_co, _Action, _Reward]]):
    def __init__(
        self,
        dataset: OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward],
        batch_size: int,
        empty_batch_interval: int = 0,
        **kwargs
    ):
        kwargs.update(batch_size=batch_size, num_workers=0, collate_fn=list)
        super().__init__(dataset=dataset, **kwargs)
        self.dataset: OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward]
        self.empty_batch_interval = empty_batch_interval
        self.env = dataset

    def __iter__(self):
        # NOTE: Testing this out, seems to work so far. However, not sure if there is some kind of
        # buffer that delays stuff, so I'll probably just do it myself just to be sure.

        # IDEA: I just had a terribly stupid idea, that could fix the PL bug!
        # Give out an empty batch between every other batch. That way, the `with_is_last` bug
        # doesn't happen!

        # return super().__iter__()

        assert self.batch_size is not None
        episodes_so_far = 0
        batches_so_far = 0
        dataset_iterator = iter(self.dataset)

        buffer = []
        while True:
            try:
                episode = next(dataset_iterator)
            except StopIteration:
                break
            buffer.append(episode)
            episodes_so_far += 1

            if len(buffer) < self.batch_size:
                continue

            # Buffer is now full.
            new_policy = yield buffer
            if new_policy is not None:
                self.dataset.send(new_policy)

            if (
                self.empty_batch_interval
                and batches_so_far > 0
                and batches_so_far % self.empty_batch_interval == 0
            ):
                # STUPID idea that might just work: Yield an empty batch here, so that the model
                # consuming this performs an update, and so the next batch can use the updated
                # policy, instead of yielding a batch that is "stale".
                new_policy = yield []
                if new_policy is not None:
                    self.dataset.send(new_policy)

            buffer = []
            batches_so_far += 1

        if len(buffer) == self.batch_size or (
            0 < len(buffer) < self.batch_size and not self.drop_last
        ):
            new_policy = yield buffer
            if new_policy is not None:
                self.dataset.send(new_policy)

        # for batch_index in itertools.count():
        #     buffer = []
        #     for i, episode in zip(range(self.batch_size), self.dataset):
        #         buffer.append(episode)
        #     new_policy = yield buffer

        #     if new_policy is not None:
        #         self.dataset.send(new_policy)

        #     if (
        #         self.empty_batch_interval
        #         and batch_index > 0
        #         and batch_index % self.empty_batch_interval == 0
        #     ):
        #         # STUPID idea that might just work: Yield an empty batch here, so that the model
        #         # consuming this performs an update, and so the next batch can use the updated
        #         # policy, instead of yielding a batch that is "stale".
        #         new_policy = yield []
        #         if new_policy is not None:
        #             self.dataset.send(new_policy)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        return self.dataset.send(new_policy)
