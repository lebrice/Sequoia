""" Cute little dataclass that is used to describe a given type of Metrics.

This is a bit like the Metrics from pytorch-lightning, but seems easier to use,
as far as I know. Also totally transferable between gpus etc. (Haven't used
the metrics from PL much yet, to be honest).
"""
from collections import OrderedDict
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from utils.logging_utils import cleanup
from utils.serialization import Serializable


@dataclass
class Metrics(Serializable):
    # This field isn't used in comparisons between Metrics.
    n_samples: int = field(default=0, compare=False)

    
    # TODO: Refactor this to take any kwargs, and then let each metric type
    # specify its own InitVars.
    
    @torch.no_grad()
    def __post_init__(self, **tensors):
        """Creates metrics given `y_pred` and `y`.

        NOTE: Doesn't use `x` and `h_x` for now.

        Args:
            x (Tensor, optional): The input Tensor. Defaults to None.
            h_x (Tensor, optional): The hidden representation for x. Defaults to None.
            y_pred (Tensor, optional): The predicted label. Defaults to None.
            y (Tensor, optional): The true label. Defaults to None.
        """
        # get the batch size:
        for tensor in tensors.values():
            if isinstance(tensor, (np.ndarray, Tensor)):
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

    def __mul__(self, factor: Union[float, Tensor]) -> "Loss":
        # By default, multiplying or dividing a Metrics object doesn't change
        # anything about it.
        return self

    def __rmul__(self, factor: Union[float, Tensor]) -> "Loss":
        # Reverse-order multiply, used to do b * a when a * b returns
        # NotImplemented.
        return self.__mul__(factor)

    def __truediv__(self, coefficient: Union[float, Tensor]) -> "Metrics":
        # By default, multiplying or dividing a Metrics object doesn't change
        # anything about it. 
        return self

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
        if verbose:
            return {"n_samples": self.n_samples}
        return {}

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        return OrderedDict()

    def numpy(self):
        """Returns a new object with all the tensor fields converted to numpy arrays."""
        def to_numpy(val: Any):
            if isinstance(val, Tensor):
                return val.detach().cpu().numpy()
            if isinstance(val, (list, tuple)):
                return np.array(val)
            return val
        return type(self)(**{
            name: to_numpy(val) for name, val in self.items()
        })
