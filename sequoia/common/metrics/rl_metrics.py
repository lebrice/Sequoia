from dataclasses import dataclass, field
from typing import Any, Dict, Union

from .metrics import Metrics


@dataclass
class EpisodeMetrics(Metrics):
    """Metrics for Episodes in RL.

    n_samples is the number of stored episodes.
    """

    n_samples: int = field(default=1, compare=False)
    # The average reward per episode.
    mean_episode_reward: float = 0.0
    # The average length of each episode.
    mean_episode_length: float = 0

    @property
    def n_episodes(self) -> int:
        return self.n_samples

    @property
    def objective_name(self) -> str:
        """Returns the name to be associated with the objective of this class.

        Returns
        -------
        str
            The name associated with the objective.
        """
        return "Mean Reward per Episode"

    @property
    def mean_reward_per_step(self) -> float:
        return self.mean_episode_reward / self.mean_episode_length

    def __add__(self, other: Union["EpisodeMetrics", Any]):
        if isinstance(other, (int, float)) and other == 0:
            # This makes `sum(list_of_metrics)` work!.
            return self
        if isinstance(other, Metrics) and other == Metrics():
            return self
        if not isinstance(other, EpisodeMetrics):
            return NotImplemented

        other: EpisodeMetrics
        other_total_reward = other.mean_episode_reward * other.n_samples
        other_total_length = other.mean_episode_length * other.n_samples
        self_total_reward = self.mean_episode_reward * self.n_samples
        self_total_length = self.mean_episode_length * self.n_samples

        new_n_samples = self.n_samples + other.n_samples
        new_mean_reward = (self_total_reward + other_total_reward) / new_n_samples
        new_mean_length = (self_total_length + other_total_length) / new_n_samples

        return EpisodeMetrics(
            n_samples=new_n_samples,
            mean_episode_reward=new_mean_reward,
            mean_episode_length=new_mean_length,
        )

    @property
    def total_reward(self) -> float:
        return self.n_episodes * self.mean_episode_reward

    @property
    def total_steps(self) -> int:
        return round(self.n_episodes * self.mean_episode_length)

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        return self.to_log_dict()

    @property
    def objective(self) -> float:
        return self.mean_episode_reward

    def to_log_dict(self, verbose: bool = False):
        log_dict = {
            "Episodes": self.n_episodes,
            "Mean reward per episode": self.mean_episode_reward,
            "Mean reward per step": self.mean_reward_per_step,
        }
        if verbose:
            log_dict.update(
                {
                    "Total steps": int(self.total_steps),
                    "Total reward": int(self.total_reward),
                    "Mean episode length": float(self.mean_episode_length),
                }
            )
        return log_dict

    @property
    def episodes(self) -> int:
        return self.n_samples

    @property
    def mean_reward_per_episode(self) -> float:
        return self.mean_episode_reward


# @dataclass
# class RLMetrics(Metrics):
#     episodes: List[EpisodeMetrics] = field(default_factory=list, repr=False)

#     average_episode_length: int = field(default=0)
#     average_episode_reward: float = field(default=0.)

#     def __post_init__(self):
#         if self.episodes:
#             self.n_samples = len(self.episodes)
#             self.average_episode_length = sum(ep.episode_length for ep in self.episodes) / self.n_samples
#             self.average_episode_reward = sum(ep.total_reward for ep in self.episodes) / self.n_samples

#     def __add__(self, other: Union["RLMetrics", EpisodeMetrics, Any]) -> "RLMetrics":
#         if isinstance(other, RLMetrics):
#             return RLMetrics(
#                 episodes = self.episodes + other.episodes,
#             )
#         if isinstance(other, EpisodeMetrics):
#             self.episodes.append(other)
#             return self
#         return NotImplemented

#     def to_pbar_message(self) -> Dict[str, Union[str, float]]:
#         log_dict = self.to_log_dict()
#         # Rename "n_samples" to "episodes":
#         log_dict["episodes"] = log_dict.pop("n_samples")
#         return log_dict


@dataclass
class GradientUsageMetric(Metrics):
    """Small Metrics to report the fraction of gradients that were used vs
    'wasted', when using batch_size > 1.
    """

    used_gradients: int = 0
    wasted_gradients: int = 0
    used_gradients_fraction: float = 0.0

    def __post_init__(self):
        self.n_samples = self.used_gradients + self.wasted_gradients
        if self.n_samples:
            self.used_gradients_fraction = self.used_gradients / self.n_samples

    def __add__(self, other: Union["GradientUsageMetric", Any]) -> "GradientUsageMetric":
        if not isinstance(other, GradientUsageMetric):
            return NotImplemented
        return GradientUsageMetric(
            used_gradients=self.used_gradients + other.used_gradients,
            wasted_gradients=self.wasted_gradients + other.wasted_gradients,
        )

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        return {"used_fraction": self.used_gradients_fraction}
