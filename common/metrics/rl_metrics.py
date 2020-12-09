from dataclasses import dataclass, field
from typing import List, Optional, Union, Any, ClassVar

from simple_parsing import list_field

from .metrics import Metrics
import numpy as np

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
        self.n_samples = len(self.rewards)
        self.episode_length = len(self.rewards)
        self.total_reward = sum(self.rewards)
        self.mean_reward = self.total_reward / self.episode_length
        
        # IDEA: Maybe use the 'variance' in actions/rewards as a metric?

    
@dataclass
class RLMetrics(Metrics):
    decode_into_subclasses: ClassVar[bool] = True
    # Dict mapping from starting step to episode.
    episodes: List[EpisodeMetrics] = field(default_factory=list, repr=False)
    
    average_episode_length: int =    field(default=0.)
    average_episode_reward: float =  field(default=0.)

    def __post_init__(self):
        if self.episodes:
            self.n_samples = len(self.episodes)
            self.average_episode_length = np.mean([ep.episode_length for ep in self.episodes])
            self.average_episode_reward = np.mean([ep.total_reward for ep in self.episodes])
        
    def __add__(self, other: Union["RLMetrics", EpisodeMetrics, Any]) -> "RLMetrics":
        if isinstance(other, RLMetrics):
            return RLMetrics(
                episodes = self.episodes + other.episodes,
            )
        if isinstance(other, EpisodeMetrics):
            self.episodes.append(other)
            return self
        return NotImplemented
    
# m = RLMetrics(episodes=[EpisodeMetrics(rewards=[1, 2, 3])])
# metrics = Metrics.from_dict(m.to_dict(), drop_extra_fields=False)
# assert False, metrics