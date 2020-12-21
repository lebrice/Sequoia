from dataclasses import dataclass, field, fields
from typing import List, Optional, Union, Any, ClassVar, Dict

from simple_parsing import list_field
import numpy as np

from sequoia.utils.generic_functions import detach

from .metrics import Metrics


@dataclass
class EpisodeMetrics(Metrics):
    """ Metrics for an Episode in RL. """
    rewards: List[float] = field(default_factory=list, repr=False)
    # actions: List[int] = field(default_factory=list, repr=False)
    # log_probs: List[Sequence[float]] = field(default_factory=list, repr=False)
    
    # Step at which the episode started.
    start_step: Optional[int] = None
    
    episode_length: int = field(default=0)
    total_reward: float = field(default=0.)
    mean_reward: float  = field(default=0.)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.rewards = detach(self.rewards)
        self.n_samples = len(self.rewards)
        self.episode_length = len(self.rewards)
        self.total_reward = sum(self.rewards)
        self.mean_reward = self.total_reward / self.episode_length
        
        # IDEA: Maybe use the 'variance' in actions/rewards as a metric?
 
    def __add__(self, other):
        # IDEA: Adding two EpisodeMetrics together gives you a RLMetrics??
        # if isinstance(other, EpisodeMetrics):
        #     return RLMetrics(episodes=[self, other])
        return NotImplemented

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        log_dict = self.to_log_dict()
        log_dict.pop("rewards")
        return log_dict
    # def to_log_dict(self):


@dataclass
class RLMetrics(Metrics):
    # Dict mapping from starting step to episode.
    episodes: List[EpisodeMetrics] = field(default_factory=list, repr=False)
    
    average_episode_length: int = field(default=0)
    average_episode_reward: float = field(default=0.)

    def __post_init__(self):
        if self.episodes:
            self.n_samples = len(self.episodes)
            self.average_episode_length = sum(ep.episode_length for ep in self.episodes) / self.n_samples
            self.average_episode_reward = sum(ep.total_reward for ep in self.episodes) / self.n_samples
        
    def __add__(self, other: Union["RLMetrics", EpisodeMetrics, Any]) -> "RLMetrics":
        if isinstance(other, RLMetrics):
            return RLMetrics(
                episodes = self.episodes + other.episodes,
            )
        if isinstance(other, EpisodeMetrics):
            self.episodes.append(other)
            return self
        return NotImplemented

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        log_dict = self.to_log_dict()
        # Rename "n_samples" to "episodes":
        log_dict["episodes"] = log_dict.pop("n_samples")
        return log_dict


@dataclass
class GradientUsageMetric(Metrics):
    """ Small Metrics to report the fraction of gradients that were used vs
    'wasted', when using batch_size > 1.
    """
    used_gradients: int = 0
    wasted_gradients: int = 0
    used_gradients_fraction: float = 0.
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
        return {
            "used_gradients_fraction": self.used_gradients_fraction
        }