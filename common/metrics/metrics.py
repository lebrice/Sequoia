from collections import OrderedDict
from dataclasses import InitVar, dataclass
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from utils.json_utils import Serializable
from utils.logging_utils import cleanup


@dataclass
class Metrics(Serializable):
    n_samples: int = 0

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None

    @torch.no_grad()
    def __post_init__(self,
                      x: Tensor = None,
                      h_x: Tensor = None,
                      y_pred: Tensor = None,
                      y: Tensor = None):
        """Creates metrics given `y_pred` and `y`.

        NOTE: Doesn't use `x` and `h_x` for now.

        Args:
            x (Tensor, optional): The input Tensor. Defaults to None.
            h_x (Tensor, optional): The hidden representation for x. Defaults to None.
            y_pred (Tensor, optional): The predicted label. Defaults to None.
            y (Tensor, optional): The true label. Defaults to None.
        """
        # get the batch size:
        for tensor in [x, h_x, y_pred, y]:
            if tensor is not None:
                self.n_samples = tensor.shape[0]
                break

    def __add__(self, other):
        # Instances of the Metrics base class shouldn't be added together, as
        # the subclasses should implement the method. We just return the other.
        return other

    def __radd__(self, other):
        # Instances of the Metrics base class shouldn't be added together, as
        # the subclasses should implement the method. We just return the other.
        if isinstance(other, (int, float)) and other == 0.:
            return self
        return NotImplemented

    def to_log_dict(self, verbose: bool = False) -> Dict:
        """Creates a dictionary to be logged (e.g. by `wandb.log`).

        Args:
            verbose (bool, optional): Wether to include a lot of information, or
            to only log the 'essential' metrics. See the `cleanup` function for
            more info. Defaults to False.

        Returns:
            Dict: A dict containing the things to be logged.

        TODO: Maybe create a `make_plots()` method to get wandb plots from the
        metric?
        """
        log_dict = self.to_dict()
        if verbose:
            return log_dict
        return cleanup(log_dict)

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        return OrderedDict()
