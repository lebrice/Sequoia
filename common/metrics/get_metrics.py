""" Defines the get_metrics function with gives back appropriate metrics
for the given tensors.

TODO: Add more metrics! Maybe even fancy things that are based on the
hidden vectors like wasserstein distance, etc?
"""
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from utils.logging_utils import get_logger
from utils.utils import to_optional_tensor

from .classification import ClassificationMetrics
from .metrics import Metrics
from .regression import RegressionMetrics

logger = get_logger(__file__)


@torch.no_grad()
def get_metrics(y_pred: Union[Tensor, np.ndarray],
                y: Union[Tensor, np.ndarray],
                x: Union[Tensor, np.ndarray]=None,
                h_x: Union[Tensor, np.ndarray]=None) -> Optional[Metrics]:
    y = to_optional_tensor(y)
    y_pred = to_optional_tensor(y_pred)
    x = to_optional_tensor(x)
    h_x = to_optional_tensor(h_x)
    if y is not None and y_pred is not None:
        if y.shape != y_pred.shape:
            # TODO: I think this condition also works for binary classification,
            # at least when the logits have a shape[-1] == 2, but I don't know if it
            # would cause some trouble if there is a single logit, rather than 2.
            return ClassificationMetrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
        return RegressionMetrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
    return None
