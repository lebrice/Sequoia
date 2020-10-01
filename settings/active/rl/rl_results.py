from settings.base import Results
from dataclasses import dataclass

@dataclass
class RLResults(Results):
    mean_reward: float

    @property
    def objective(self) -> float:
        return self.mean_reward
