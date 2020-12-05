""" Metrics class for regression.

Gives the mean squared error between a prediction Tensor `y_pred` and the
target tensor `y`. 
"""

from dataclasses import dataclass, InitVar
from typing import Dict, Union, Any, Optional
from functools import total_ordering

import torch
import torch.nn.functional as functional
from torch import Tensor

from utils.logging_utils import get_logger
from utils.logging_utils import cleanup
from .metrics import Metrics

logger = get_logger(__file__)

@total_ordering
@dataclass
class RegressionMetrics(Metrics):
    """TODO: Use this in the RL settings! """
    mse: Tensor = 0.  # type: ignore
    l1_error: Tensor = 0.  # type: ignore

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None
    
    def __post_init__(self,
                      x: Tensor=None,
                      h_x: Tensor=None,
                      y_pred: Tensor=None,
                      y: Tensor=None):
        super().__post_init__(x=x, h_x=h_x, y_pred=y_pred, y=y)
        if y_pred is not None and y is not None:
            if y.shape != y_pred.shape:
                logger.warning(UserWarning(
                    f"Shapes aren't the same! (y_pred.shape={y_pred.shape}, "
                    f"y.shape={y.shape}"
                ))
            else:
                self.mse = functional.mse_loss(y_pred, y)
                self.l1_error = functional.l1_loss(y_pred, y)

        self.mse = torch.as_tensor(self.mse)
        self.l1_error = torch.as_tensor(self.l1_error)

    @property
    def objective(self) -> float:
        return float(self.mse)

    def __add__(self, other: "RegressionMetrics") -> "RegressionMetrics":
        # NOTE: Creates new tensors, and links them to the previous ones by
        # addition so the grads are linked.
        if self.mse is not None:
            mse = self.mse.clone()
        if other.mse is not None:
            mse = other.mse.clone()
        else:
            mse = torch.zeros(1)

        if self.l1_error is not None:
            l1_error = self.l1_error.clone()
        if other.l1_error is not None:
            l1_error = other.l1_error.clone()
        else:
            l1_error = torch.zeros(1)

        return RegressionMetrics(
            n_samples=self.n_samples + other.n_samples,
            mse=mse,
            l1_error=l1_error,
        )

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["mse"] = float(self.mse.item())
        message["l1_error"] = float(self.l1_error.item())
        return message
    
    def to_log_dict(self, verbose=False):
        log_dict = super().to_log_dict(verbose=verbose)
        log_dict["mse"] = self.mse
        log_dict["l1_error"] = self.l1_error
        return log_dict
    
    def __mul__(self, factor: Union[float, Tensor]) -> "Loss":
        # Multiplying a 'RegressionMetrics' object multiplies its 'mse'.
        return RegressionMetrics(
            n_samples=self.n_samples,
            mse=self.mse * factor,
            l1_error=self.l1_error * factor,
        )

    def __rmul__(self, factor: Union[float, Tensor]) -> "Loss":
        # Reverse-order multiply, used to do b * a when a * b returns
        # NotImplemented.
        return self.__mul__(factor)

    def __truediv__(self, coefficient: Union[float, Tensor]) -> "RegressionMetrics":
        # Dividing a RegressionMetrics object divides its mean squared error.
        return RegressionMetrics(
            n_samples=self.n_samples,
            mse=self.mse / coefficient,
            l1_error=self.l1_error / coefficient,
        )
    
    def __lt__(self, other: Union["RegressionMetrics", Any]) -> bool:
        if isinstance(other, RegressionMetrics):
            return self.mse < other.mse
        return NotImplemented

    def __ge__(self, other: Union["RegressionMetrics", Any]) -> bool:
        if isinstance(other, RegressionMetrics):
            return self.mse >= other.mse
        return NotImplemented
