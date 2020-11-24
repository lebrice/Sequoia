from dataclasses import dataclass
from typing import Optional, TypeVar, Any

from simple_parsing.helpers.flatten import FlattenedAccess
from torch import Tensor, LongTensor

import numpy as np
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
    def actions(self) -> Tensor:
        return self.y_pred

    @property
    def actions_np(self) -> np.ndarray:
        """ Returns the prediction/action as a numpy array. """
        if isinstance(self.y_pred, Tensor):
            return self.y_pred.detach().cpu().numpy()
        return np.asarray(self.y_pred)

    @property
    def predictions(self) -> Tensor:
        return self.y_pred


@dataclass(frozen=True)
class Rewards(Batch):
    """ A batch of "rewards" coming from an Environment.

    For example, in a supervised setting, this would be the true labels, while
    in an RL setting, this would be the 'reward' for a state-action pair.
    
    TODO: Maybe add the task labels as a part of the 'Reward', to help with the
    training of task-inference methods later on when we add those.
    """
    y: Optional[Tensor]

    @property
    def labels(self) -> Tensor:
        return self.y
    
    @property
    def reward(self) -> Tensor:
        return self.y

ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
