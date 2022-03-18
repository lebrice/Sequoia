from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

import gym
from gym.wrappers import RecordEpisodeStatistics
from matplotlib import pyplot as plt
from simple_parsing.helpers import choice
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sequoia import Results
from sequoia.settings.base import Setting

try:
    import d3rlpy
except ImportError as err:
    raise RuntimeError(f"You need to have `d3rlpy` installed to use these methods.") from err


@dataclass
class OfflineRLResults(Results):

    # TODO: Write these methods
    def summary(self) -> str:
        return f"Offline RL results: {self.objective_name} = {self.objective}"

    def make_plots(self) -> Dict[str, plt.Figure]:
        return {}

    def to_log_dict(self, verbose: bool = False) -> Dict[str, Any]:
        return {self.objective_name: self.objective}

    # Metrics from online testing
    test_rewards: list
    test_episode_length: list
    test_episode_count: list

    objective_name: ClassVar[str] = "Average Reward"

    @property
    def objective(self):
        return sum(self.test_rewards) / len(self.test_rewards)


# Offline datasets from d3rlpy (not including atari)
offline_datasets_from_d3rlpy = {
    "cartpole-replay",
    "cartpole-random",
    "pendulum-replay",
    "pendulum-random",
    "hopper",
    "halfcheetah",
    "walker",
    "ant",
}

# Offline atari datasets from d3rlpy
offline_atari_datasets_from_d3rlpy = set(d3rlpy.datasets.ATARI_GAMES)


@dataclass
class OfflineRLSetting(Setting):

    # A list of available offline rl datasets
    available_datasets: ClassVar[List[str]] = list(offline_datasets_from_d3rlpy) + list(
        offline_atari_datasets_from_d3rlpy
    )

    # choice of dataset for the current setting
    dataset: str = choice(available_datasets, default="cartpole-replay")

    # size of validation set
    val_size: float = 0.2

    # mask for control bootstrapping
    create_mask: bool = False
    mask_size: int = 1

    def __post_init__(self):
        # Load d3rlpy offline dataset
        if (
            self.dataset in offline_datasets_from_d3rlpy
            or self.dataset in offline_atari_datasets_from_d3rlpy
        ):
            mdp_dataset, self.env = d3rlpy.datasets.get_dataset(
                self.dataset, self.create_mask, self.mask_size
            )
            self.train_dataset, self.valid_dataset = train_test_split(
                mdp_dataset, test_size=self.val_size
            )

        # Load other dataset types here
        else:
            raise NotImplementedError

    def train_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size: int = None) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=batch_size)

    def test(self, method, test_env: gym.Env):
        """
        Test self.algo on given test_env for self.test_steps iterations
        """
        test_env = RecordEpisodeStatistics(test_env)

        obs = test_env.reset()
        for _ in range(method.test_steps):
            obs, reward, done, info = test_env.step(
                method.get_actions(obs, action_space=test_env.action_space)
            )
            if done:
                break
        test_env.close()

        return test_env.episode_returns, test_env.episode_lengths, test_env.episode_count

    def apply(self, method) -> OfflineRLResults:
        method.configure(self)

        method.fit(train_env=self.train_dataset, valid_env=self.valid_dataset)

        # Test
        test_rewards, test_episode_length, test_episode_count = self.test(method, self.env)
        return OfflineRLResults(
            test_rewards=test_rewards,
            test_episode_length=test_episode_length,
            test_episode_count=test_episode_count,
        )
