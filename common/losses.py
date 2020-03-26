from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from utils.utils import add_prefix

from .metrics import ClassificationMetrics, Metrics, get_metrics


def add_dicts(d1: Dict, d2: Dict, add_values=True) -> Dict:
    result = d1.copy()
    for key, v2 in d2.items():
        if isinstance(v2, dict):
            result[key] = add_dicts(d1[key], v2)
        elif not add_values:
            result[key] = v2
        elif key not in d1:
            result[key] = v2
        else:
            result[key] = d1[key] + v2
    return result


@dataclass
class LossInfo:
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    name: str
    coefficient: Union[float, Tensor] = 1.0
    total_loss: Tensor = 0.  # type: ignore
    losses:  Dict[str, Union[Tensor, "LossInfo"]] = field(default_factory=OrderedDict)
    tensors: Dict[str, Tensor] = field(default_factory=OrderedDict, repr=False)
    metrics: Dict[str, Metrics] = field(default_factory=OrderedDict)

    def __add__(self, other: "LossInfo") -> "LossInfo":
        """Adds two LossInfo instances together.
        
        Adds the losses, total loss and metrics. Overwrites the tensors.
        Keeps the name of the first one. This is useful when doing something
        like:
        
        ```
        total_loss = LossInfo("Test")
        for x, y in dataloader:
            total_loss += model.get_loss(x=x, y=y)
        ```      
        
        Returns
        -------
        LossInfo
            The merged/summed up LossInfo.
        """
        name = self.name
        total_loss = self.total_loss + other.total_loss
        
        losses  = add_dicts(self.losses, other.losses, add_values=True)
        # Keep the total loss of the other LossInfo in the `losses` dict.
        if other.name not in losses:
            losses[other.name] = other.total_loss
        
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        metrics = add_dicts(self.metrics, other.metrics, add_values=True)
        return LossInfo(
            name=name,
            coefficient=self.coefficient,
            total_loss=total_loss,
            losses=losses,
            tensors=tensors,
            metrics=metrics,
        )

    def __mul__(self, coefficient: Union[float,Tensor]) -> "LossInfo":
        """ Scale each loss tensor by `coefficient`.

        Returns
        -------
        LossInfo
            returns a scaled LossInfo instance.
        """
        return LossInfo(
            name=self.name,
            coefficient=self.coefficient * coefficient,
            total_loss=self.total_loss * coefficient,
            losses=OrderedDict([
                (k, value * coefficient) for k, value in self.losses.items()
            ]),
            metrics=self.metrics,
            tensors=self.tensors,
        )

    def add_prefix(self, prefix: str) -> None:
        self.losses = add_prefix(self.losses, prefix)
        self.tensors = add_prefix(self.tensors, prefix)
        self.metrics = add_prefix(self.metrics, prefix)

    @property
    def unscaled_losses(self):
        return OrderedDict([
            (k, value / self.coefficient) for k, value in self.losses.items()
        ])

    def to_log_dict(self) -> Dict:
        log_dict: Dict[str, Any] = OrderedDict()
        log_dict["name"] = self.name
        log_dict["coefficient"] = self.coefficient.item() if isinstance(self.coefficient, Tensor) else self.coefficient
        log_dict['total_loss'] = self.total_loss.item()

        losses: Dict[str, Union[float, List, Dict]] = OrderedDict()
        for loss_name, loss in self.losses.items():
            if isinstance(loss, Tensor):
                losses[loss_name] = loss.item() if loss.numel() == 1 else loss.tolist()
            elif isinstance(loss, LossInfo):
                losses[loss_name] = loss.to_log_dict()
        log_dict["losses"] = losses

        metrics: Dict[str, Dict] = OrderedDict()
        for name, metric in self.metrics.items():
            metrics[name] = metric.to_log_dict()
        log_dict["metrics"] = metrics

        # return add_prefix(log_dict, self.name)
        return log_dict
