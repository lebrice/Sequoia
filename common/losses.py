from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Union, Optional

import torch
from torch import Tensor
from .metrics import Metrics

def add_dicts(d1: Dict, d2: Dict, add_values=True):
    result = d1.copy()
    for key, v2 in d2.items():
        if add_values:
            result[key] = result.get(key, 0) + v2
        else:
            result[key] = v2
    return result

@dataclass
class LossInfo:
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    total_loss: Tensor = 0.  # type: ignore
    losses: Dict[str, Tensor] = field(default_factory=OrderedDict)
    tensors: Dict[str, Tensor] = field(default_factory=OrderedDict, repr=False)
    metrics: Metrics = field(default_factory=Metrics)

    def __add__(self, other: "LossInfo") -> "LossInfo":
        total_loss = self.total_loss + other.total_loss
        losses = add_dicts(self.losses, other.losses)
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        metrics = self.metrics + other.metrics     
        return LossInfo(
            total_loss=total_loss,
            losses=losses,
            tensors=tensors,
            metrics=metrics,
        )

    def scale_by(self, coefficient: Union[float,Tensor]) -> "LossInfo":
        """ Scale each loss tensor by `coefficient` and add the scaled losses
        to the dict of losses.
        
        Returns
        -------
        LossInfo
            returns `self`.
        """
        if self.total_loss == 0 and self.losses:
            self.total_loss = torch.sum(self.losses.values())  # type: ignore
        self.losses["total"] = self.total_loss
        
        self.total_loss = self.total_loss * coefficient

        scaled_losses: Dict[str, Tensor] = {}
        for loss_name, loss_tensor in self.losses.items():
            scaled_losses[f"{loss_name}_scaled"] = coefficient * loss_tensor
        self.losses.update(scaled_losses)
        return self

    def add_prefix(self, prefix: str) -> None:
        prepend(self.losses, prefix)
        prepend(self.tensors, prefix)
    
    def to_log_dict(self) -> Dict:
        return {
            'total_loss': self.total_loss.item(), # if isinstance(self.total_loss, torch.Tensor) else self.total_loss,
            **{k: v.item() for (k, v) in self.losses.items()},
            **self.metrics.to_log_dict()
        }


def prepend(d: Dict, prefix: str) -> None:
    for key in list(d.keys()):
        if not key.startswith(prefix):
            value = d.pop(key)
            d[f"{prefix}.{key}"] = value
