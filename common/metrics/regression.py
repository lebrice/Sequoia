""" Metrics class for regression.

Gives the mean squared error between a prediction Tensor `y_pred` and the
target tensor `y`. 
"""

from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn.functional as functional
from torch import Tensor

from utils.logging_utils import get_logger
from utils.logging_utils import cleanup
from .metrics import Metrics

logger = get_logger(__file__)

@dataclass
class RegressionMetrics(Metrics):
    mse: Tensor = 0.  # type: ignore

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

    def __add__(self, other: "RegressionMetrics") -> "RegressionMetrics":
        mse = torch.zeros_like(
            self.mse if self.mse is not None else
            other.mse if other.mse is not None else
            torch.zeros(1)
        )
        if self.mse is not None:
            mse = mse + self.mse
        if other.mse is not None:
            mse = mse + other.mse
        return RegressionMetrics(
            n_samples=self.n_samples + other.n_samples,
            mse=mse,
        )

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["mse"] = float(self.mse.item())
        return message
