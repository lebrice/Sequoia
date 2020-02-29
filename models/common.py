from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Union

import torch
from torch import Tensor


@dataclass
class LossInfo:
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    total_loss: Tensor = field(default_factory=lambda: torch.zeros(1).clone())
    losses: Dict[str, Tensor] = field(default_factory=OrderedDict)
    tensors: Dict[str, Tensor] = field(default_factory=OrderedDict)
    metrics: Dict[str, Any] = field(default_factory=OrderedDict)

    def __iadd__(self, other: "LossInfo") -> "LossInfo":
        self.total_loss += other.total_loss
        self.losses.update(other.losses)
        self.tensors.update(other.tensors)
        self.metrics.update(other.metrics)
        return self

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
        prepend(self.metrics, prefix)

def prepend(d: Dict, prefix: str) -> None:
    for key in list(d.keys()):
        if not key.startswith(prefix):
            value = d.pop(key)
            d[f"{prefix}.{key}"] = value