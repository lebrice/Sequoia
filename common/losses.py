import itertools
import logging
from collections import OrderedDict
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TypeVar

import torch
from torch import Tensor

from utils.json_utils import JsonSerializable
from utils.utils import add_dicts, add_prefix

from .metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                      get_metrics)

logger = logging.getLogger(__file__)


@dataclass
class LossInfo(JsonSerializable):
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
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.requires_grad:
                self.tensors[name] = tensor.detach()
        if isinstance(self.total_loss, list):
            self.total_loss = torch.as_tensor(self.total_loss)
        
        for name, loss in self.losses.items():
            if isinstance(loss, dict):
                self.losses[name] = LossInfo.from_dict(loss)

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

    def to_log_dict(self, verbose: bool=False) -> Dict[str, Union[str, float, Dict]]:
        log_dict: Dict[str, Union[str, float, Dict]] = OrderedDict()
        # Log the total loss
        log_dict["loss"] = float(self.total_loss)
        
        # Log the metrics
        metrics: Dict[str, Dict] = OrderedDict()
        for metric_name, metric in self.metrics.items():
            metric_log_dict = metric.to_log_dict(verbose=verbose)
            if metric_name not in metrics:
                metrics[metric_name] = OrderedDict()
            metrics[metric_name].update(metric_log_dict)
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

    def to_dict(self):
        return self.to_log_dict(verbose=False)
    
    def drop_tensors(self) -> None:
        self.tensors.clear()
        for n, loss in self.losses.items():
            loss.drop_tensors()


@dataclass
class TrainValidLosses(JsonSerializable):
    """ Helper class to store the train and valid losses during training. """
    train_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)
    valid_losses: Dict[int, LossInfo] = field(default_factory=OrderedDict)

    def __iadd__(self, other: Union["TrainValidLosses", Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]]) -> "TrainValidLosses":
        if isinstance(other, TrainValidLosses):
            self.train_losses.update(other.train_losses)
            self.valid_losses.update(other.valid_losses)
        elif isinstance(other, tuple):
            self.train_losses.update(other[0])
            self.valid_losses.update(other[1])
        else:
            return NotImplemented
        self.drop_tensors()
        return self
    
    def __setitem__(self, index: int, value: Tuple[LossInfo, LossInfo]) -> None:
        self.train_losses[index] = value[0].detach()
        self.valid_losses[index] = value[1].detach()

    def __getitem__(self, index: int) -> Tuple[LossInfo, LossInfo]:
        return (
            self.train_losses[index],
            self.valid_losses[index]
        )

    def all_loss_names(self) -> Set[str]:
        all_loss_names: Set[str] = set()
        for loss_info in itertools.chain(self.train_losses.values(), 
                                         self.valid_losses.values()):
            all_loss_names.update(loss_info.losses)
        return all_loss_names
    
    def latest_step(self) -> int:
        """Returns the latest global_step in the dicts."""
        return max(itertools.chain(self.train_losses, self.valid_losses), default=0)

    def keep_up_to_step(self, step: int) -> None:
        """Keeps only the losses up to step `step`.

        Args:
            step (int): the maximum step (inclusive) to keep.
        """
        for k in filter(lambda k: k > step, list(self.train_losses.keys())):
            self.train_losses.pop(k)
        for k in filter(lambda k: k > step, list(self.valid_losses.keys())):
            self.valid_losses.pop(k)

    def add_step(self, offset: int):
        """Adds the value of `offset` to all the keys in the dictionary.
        Args:
            offset (int): A value to add to all the keys.
        """
        new_train_losses: Dict[int, LossInfo] = OrderedDict()
        new_valid_losses: Dict[int, LossInfo] = OrderedDict()
        for k in list(self.train_losses.keys()):
            new_train_losses[k + offset] = self.train_losses.pop(k)
        for k in list(self.valid_losses.keys()):
            new_valid_losses[k + offset] = self.valid_losses.pop(k)
        self.train_losses = new_train_losses
        self.valid_losses = new_valid_losses

    def drop_tensors(self) -> None:
        for l in self.train_losses.values():
            l.drop_tensors()
        for l in self.valid_losses.values():
            l.drop_tensors()

# @encode.register
# def encode_losses(obj: TrainValidLosses) -> Dict:
#     train_losses_dict = OrderedDict((k, encode(v)) for k, v in obj.train_losses.items())
#     valid_losses_dict = OrderedDict((k, encode(v)) for k, v in obj.valid_losses.items())
#     return {
#         "train_losses": train_losses_dict,
#         "valid_losses": valid_losses_dict,
#     }
