from __future__ import annotations
from torch.utils.data import DataLoader, ChainDataset
from torch.utils.data.dataloader import (
    Dataset,
    T_co,
    Sampler,
    _collate_fn_t,
    _worker_init_fn_t,
)
from typing import Optional, Sequence, Union, Iterator, Optional

from torch.utils.data.dataset import IterableDataset
from sequoia.common.episode_collector.policy import Policy
from sequoia.common.typed_gym import (
    _Env,
    _VectorEnv,
    _Observation_co,
    _Action,
    _Reward,
    _Reward,
)
from .episode import Episode
from .episode_collector import EpisodeCollector, PolicyUpdateStrategy, do_nothing_strategy, redo_forward_pass_strategy
import gym


def make_env_loader(
    env: _Env[_Observation_co, _Action, _Reward],
    policy: Policy[_Observation_co, _Action],
    max_episodes: int = None,
    max_steps: int = None,
    seed: int = None,
    what_to_do_after_update: PolicyUpdateStrategy = redo_forward_pass_strategy,
) -> OnPolicyEpisodeLoader[_Observation_co, _Action, _Reward]:
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    dataset = OnPolicyEpisodeDataset(
        env=env,
        policy=policy,
        max_iter_steps=max_steps,
        max_iter_episodes=max_episodes,
        what_to_do_after_update=what_to_do_after_update,
    )
    loader = OnPolicyEpisodeLoader(dataset=dataset)
    return loader


class OnPolicyEpisodeDataset(
    gym.Wrapper, IterableDataset[Episode[_Observation_co, _Action, _Reward]]
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

    def __iter__(self) -> Iterator[Episode[_Observation_co, _Action, _Reward]]:
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
            "OnPolicyEpisodeDataset[Episode[_Observation_co, _Action, _Reward]]",
        ],
    ) -> "OnPolicyEpisodeDataset[Episode[_Observation_co, _Action, _Reward]]":
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


class OnPolicyEpisodeLoader(DataLoader[Episode[_Observation_co, _Action, _Reward]]):
    def __init__(
        self, dataset: OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward], batch_size: int, **kwargs
    ):
        kwargs.update(batch_size=batch_size, num_workers=0, collate_fn=None)
        super().__init__(dataset=dataset, **kwargs)
        self.dataset: OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward]
        self.env = dataset

    def __iter__(self):
        return super().__iter__()
        # return iter(self.dataset)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        return self.dataset.send(new_policy)
