from typing import Union
from torch import Tensor, LongTensor
from sequoia.common.batch import Batch
from dataclasses import dataclass
from sequoia.settings.base import Actions


@dataclass(frozen=True)
class ClassificationActions(Actions):
    """ Typed dict-like class that represents the 'forward pass'/output of a
    classification head, which correspond to the 'actions' to be sent to the
    environment, in the general formulation.
    """
    y_pred: Union[LongTensor, Tensor]
    logits: Tensor

    @property
    def action(self) -> LongTensor:
        return self.y_pred
    
    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.logits[:, self.y_pred]

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.probabilities[self.y_pred]

    @property
    def probabilities(self) -> Tensor:
        """ Returns the normalized probabilies for each class, i.e. the
        softmax-ed version of `self.logits`.
        """
        return self.logits.softmax(-1)
