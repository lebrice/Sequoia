""" Typed object that represents the outputs of the forward pass of a model. """

from dataclasses import dataclass
from typing import Optional, TypeVar, Any

from simple_parsing.helpers.flatten import FlattenedAccess
from torch import Tensor

from sequoia.common import Batch
from sequoia.settings.base.objects import Observations, Actions, Rewards


@dataclass(frozen=True)
class ForwardPass(Batch, FlattenedAccess):
    """ Typed version of the result of a forward pass through a model.

    FlattenedAccess is pretty cool, but potentially confusing. We can get
    any attributes in the children by getting them directly on the
    parent. So if the `observation` has an `x` attribute, we can get on this
    object directly with `self.x`, and it will fetch the attribute from the
    observation.
    """
    observations: Observations
    representations: Tensor
    actions: Actions
    rewards: Optional[Rewards]

    @property
    def h_x(self) -> Any:
        return self.representations
