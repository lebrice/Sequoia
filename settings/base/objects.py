from dataclasses import dataclass
from typing import Optional, TypeVar, Any

from simple_parsing.helpers.flatten import FlattenedAccess
from torch import Tensor

from common import Batch


@dataclass(frozen=True)
class Observations(Batch):
    """ A batch of "observations" coming from an Environment. """
    x: Tensor

    @property
    def state(self) -> Tensor:
        return self.x


@dataclass(frozen=True)
class Actions(Batch):
    """ A batch of "actions" coming from an Environment.
    
    For example, in a supervised setting, this would be the predicted labels,
    while in an RL setting, this would be the next 'actions' to take in the
    Environment.
    """
    y_pred: Tensor
    
    @property
    def action(self) -> Tensor:
        return self.y_pred
    @property
    def prediction(self) -> Tensor:
        return self.y_pred


@dataclass(frozen=True)
class Rewards(Batch):
    """ A batch of "rewards" coming from an Environment.

    For example, in a supervised setting, this would be the true labels, while
    in an RL setting, this would be the 'reward' for a state-action pair.
    """
    y: Optional[Tensor]

    @property
    def labels(self) -> Tensor:
        return self.y

ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
