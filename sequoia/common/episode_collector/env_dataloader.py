from torch.utils.data import DataLoader, ChainDataset
from torch.utils.data.dataloader import Dataset, T_co, Sampler, _collate_fn_t, _worker_init_fn_t
from typing import Optional, Sequence, Union, Iterator, Optional

from torch.utils.data.dataset import IterableDataset
from sequoia.common.episode_collector.policy import Policy
from sequoia.common.typed_gym import Env, VectorEnv, Observation_co, Action, Reward, Reward
from .episode import Episode, Observation
from .episode_collector import EpisodeCollector
import gym


class EnvDataset(gym.Wrapper, IterableDataset[Episode[Observation_co, Action, Reward]]):
    def __init__(self, env: Env[Observation_co, Action, Reward], policy: Policy[Observation_co, Action]) -> None:
        super().__init__(env=env)
        self.env = env
        self._episode_generator: Optional[EpisodeCollector[Observation_co, Action, Reward]] = None
        self.policy = policy

    def __iter__(self) -> Iterator[Episode[Observation, Action, Reward]]:
        if self._episode_generator is not None:
            self._episode_generator.close()
        # note: No access to the max_steps or max_episodes args of EpisodeCollector for now.
        # Could be confusing with the max steps or episodes when interacting with the env.
        self._episode_generator = EpisodeCollector(self.env, policy=self.policy)
        return self._episode_generator

    def send(self, new_policy: Policy[Observation_co, Action]) -> None:
        self.policy = new_policy
        self._episode_generator.send(new_policy)

    def __add__(self, other: Dataset[T_co]):
        raise NotImplementedError(
            "todo: Create ChainDatasetWithSend that passes things sent to `send` to the current "
            "dataset."
        )
        return ChainDataset([self, other])


class EnvDataLoader(DataLoader[Episode[Observation_co, Action, Reward]]):
    def __init__(self, dataset: EnvDataset[Observation_co, Action, Reward], **kwargs):
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
    
    def send(self, new_policy: Policy[Observation_co, Action]) -> None:
        return self.dataset.send(new_policy)
