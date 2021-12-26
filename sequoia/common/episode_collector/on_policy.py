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
from .episode import Episode, Observation
from .episode_collector import EpisodeCollector
import gym


class OnPolicyEpisodeDataset(
    gym.Wrapper, IterableDataset[Episode[_Observation_co, _Action, _Reward]]
):
    def __init__(
        self,
        env: _Env[_Observation_co, _Action, _Reward],
        policy: Policy[_Observation_co, _Action],
    ) -> None:
        super().__init__(env=env)
        self.env = env
        self._episode_generator: Optional[
            EpisodeCollector[_Observation_co, _Action, _Reward]
        ] = None
        self.policy = policy

    def __iter__(self) -> Iterator[Episode[Observation, _Action, _Reward]]:
        if self._episode_generator is not None:
            self._episode_generator.close()
        # note: No access to the max_steps or max_episodes args of EpisodeCollector for now.
        # Could be confusing with the max steps or episodes when interacting with the env.
        self._episode_generator = EpisodeCollector(self.env, policy=self.policy)
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

        raise NotImplementedError("TODO: Think about when this is useful.")
        return type(self)(
            env=ConcatEnvsWrapper(envs=[self.env, other]), policy=self.policy
        )


class OnPolicyEpisodeLoader(DataLoader[Episode[_Observation_co, _Action, _Reward]]):
    def __init__(
        self, dataset: OnPolicyEpisodeDataset[_Observation_co, _Action, _Reward], **kwargs
    ):
        batch_size = getattr(dataset, "num_envs", None)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=None,
        )
        self.env = dataset

    def __iter__(self):
        return iter(self.dataset)

    def send(self, new_policy: Policy[_Observation_co, _Action]) -> None:
        return self.dataset.send(new_policy)
