import itertools
from collections import OrderedDict
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from utils.utils import add_prefix

from .metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                      get_metrics)


def add_dicts(d1: Dict, d2: Dict, add_values=True) -> Dict:
    result = d1.copy()
    for key, v2 in d2.items():
        if key not in d1:
            result[key] = v2
        elif isinstance(v2, dict):
            result[key] = add_dicts(d1[key], v2, add_values=add_values)
        elif not add_values:
            result[key] = v2
        else:
            result[key] = d1[key] + v2
    return result


@dataclass
class LossInfo:
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    name: str = ""
    coefficient: Union[float, Tensor] = 1.0
    total_loss: Tensor = 0.  # type: ignore
    losses:  Dict[str, "LossInfo"] = field(default_factory=OrderedDict)
    tensors: Dict[str, Tensor] = field(default_factory=OrderedDict, repr=False)
    metrics: Dict[str, Metrics] = field(default_factory=OrderedDict)

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None

    def __post_init__(self, x: Tensor=None, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None):
        if self.name and self.name not in self.metrics:
            if y_pred is not None and y is not None:
                self.metrics[self.name] = get_metrics(y_pred=y_pred, y=y)
        for name, tensor in self.tensors.items():
            if tensor.requires_grad:
                self.tensors[name] = tensor.detach()

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
        
        if self.name == other.name:
            losses  = add_dicts(self.losses, other.losses)
            metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire LossInfo
            # object into the 'losses' dict, rather than a single loss tensor.
            losses = add_dicts(self.losses, {other.name: other})
            # TODO: setting in the 'metrics' dict, we are duplicating the
            # metrics, since they now reside in the `self.metrics[other.name]`
            # and `self.losses[other.name].metrics` attributes.
            # metrics = self.metrics
            metrics = add_dicts(self.metrics, other.metrics)
        
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
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

    @property
    def unscaled_losses(self):
        return OrderedDict([
            (k, value / self.coefficient) for k, value in self.losses.items()
        ])

    def to_log_dict(self, verbose: bool=True) -> Dict[str, Union[str, float, Dict]]:
        log_dict: Dict[str, Union[str, float, Dict]] = OrderedDict()
        # Log the total loss
        log_dict["loss"] = float(self.total_loss.item())
        # Log the metrics
        metrics: Dict[str, Dict] = OrderedDict()
        for metric_name, metric in self.metrics.items():
            metric_log_dict = metric.to_log_dict() 
            metrics.update(metric_log_dict)
        log_dict["metrics"] = metrics

        tensors: Dict[str, List] = OrderedDict()
        if verbose:
            for name, tensor in self.tensors.items():
                tensors[name] = tensor.tolist()
        log_dict["tensors"] = tensors

        # Add the loss components as nested dicts, each with their own loss and metrics.
        for name, loss_info in self.losses.items():
            subloss_log_dict = loss_info.to_log_dict(verbose=verbose)
            log_dict[name] = subloss_log_dict
        return log_dict

    def to_pbar_message(self):
        """ Smaller, less-detailed version of `self.to_log_dict()` (doesn't recurse into sublosses)
        meant to be used in progressbars.
        """
        message: Dict[str, Union[str, float]] = OrderedDict()
        message["Loss"] = float(self.total_loss.item())
        for name, loss_info in self.losses.items():
            message[f"{name} Loss"] = float(loss_info.total_loss.item())
            for metric_name, metrics in loss_info.metrics.items():
                if isinstance(metrics, ClassificationMetrics):
                    message[f"{name} Acc"] = f"{metrics.accuracy:.2%}"
                elif isinstance(metrics, RegressionMetrics):
                    message[f"{name} MSE"] = float(metrics.mse.item())
        prefix = (self.name + " ") if self.name else ""
        return add_prefix(message, prefix)



@dataclass
class TrainValidLosses:
    """ Helper class to store the train and valid losses during training. """
    train_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)
    valid_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)

    def __iadd__(self, other: Union["TrainValidLosses", Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]]) -> "TrainValidLosses":
        if isinstance(other, TrainValidLosses):
            self.train_losses.update(other.train_losses)
            self.valid_losses.update(other.valid_losses)
            return self
        elif isinstance(other, tuple):
            self.train_losses.update(other[0])
            self.valid_losses.update(other[1])
            return self
        else:
            return NotImplemented

    def all_loss_names(self) -> Set[str]:
        all_loss_names: Set[str] = set()
        for loss_info in itertools.chain(self.train_losses.values(), 
                                         self.valid_losses.values()):
            all_loss_names.update(loss_info.losses)
        return all_loss_names

    def save_json(self, path: Path) -> None:
        """ TODO: save to a json file. """
        # from dataclasses import asdict
        # from utils.json_utils import to_str_dict
        # import json
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self, f=str(path.with_suffix(".pt")))
    
    @classmethod
    def try_load_json(cls, path: Path) -> Optional["TrainValidLosses"]:
        try:
            return cls.load_json(path)
        except Exception as e:
            print(f"Couldn't load from path {path}: {e}")
            return None
    
    @classmethod
    def load_json(cls, path: Path) -> Optional["TrainValidLosses"]:
        path = path.with_suffix(".pt")
        with open(path, 'rb') as f:
            return torch.load(f)
    
    def latest_step(self) -> int:
        """Returns the latest global_step in the dicts."""
        return max(itertools.chain(self.train_losses, self.valid_losses), default=0)
