import itertools
from collections import OrderedDict
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar,
                    Union)

import torch
from torch import Tensor

from simple_parsing import field, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import cleanup, get_logger
from utils.utils import add_dicts, add_prefix, unique_consecutive

from .metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                      get_metrics)

logger = get_logger(__file__)

@dataclass
class LossInfo(Serializable):
    """ Simple object to store the losses and metrics for a given task. 
    
    Used to simplify the return type of the various `get_loss` functions.    
    """
    name: str = ""
    coefficient: Union[float, Tensor] = 1.0
    total_loss: Tensor = 0.  # type: ignore
    losses:  Dict[str, "LossInfo"] = mutable_field(OrderedDict)
    tensors: Dict[str, Tensor]     = mutable_field(OrderedDict, repr=False, to_dict=False)
    metrics: Dict[str, Metrics]    = mutable_field(OrderedDict)

    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None

    def __post_init__(self, x: Tensor=None, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None):
        if self.name and self.name not in self.metrics:
            if y_pred is not None and y is not None:
                self.metrics[self.name] = get_metrics(y_pred=y_pred, y=y)

        if x is not None:
            self.tensors["x"] = x.detach()
        if h_x is not None:
            self.tensors["h_x"] = h_x.detach()
        if y_pred is not None:
            self.tensors["y_pred"] = y_pred.detach()
        if y is not None:
            self.tensors["y"] = y.detach()

        for name, tensor in self.tensors.items():
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.requires_grad:
                self.tensors[name] = tensor.detach()
            if isinstance(self.total_loss, float) and self.total_loss == 0:
                self.total_loss = tensor.new_zeros([1], requires_grad=True)

        for name, loss in self.losses.items():
            if isinstance(loss, dict):
                self.losses[name] = LossInfo.from_dict(loss)

    @property
    def metric(self) -> Optional[Metrics]:
        """Shortcut for `self.metrics[self.name]`.

        Returns:
            Optional[Metrics]: The metrics associated with this LossInfo.
        """
        return self.metrics.get(self.name)

    @property
    def accuracy(self) -> float:
        assert isinstance(self.metric, ClassificationMetrics)
        return self.metric.accuracy
    
    @property
    def mse(self) -> Tensor:
        assert isinstance(self.metric, RegressionMetrics)
        return self.metric.mse

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
            metrics = self.metrics
            # metrics = add_dicts(self.metrics, {other.name: other.metrics})
        
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return LossInfo(
            name=name,
            coefficient=self.coefficient,
            total_loss=total_loss,
            losses=losses,
            tensors=tensors,
            metrics=metrics,
        )
    
    def __iadd__(self, other: "LossInfo") -> "LossInfo":
        """Adds LossInfo to `self` in-place.
        
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
            `self`: The merged/summed up LossInfo.
        """
        self.total_loss = self.total_loss + other.total_loss
        if self.name == other.name:
            self.losses  = add_dicts(self.losses, other.losses)
            self.metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire LossInfo
            # object into the 'losses' dict, rather than a single loss tensor.
            self.losses = add_dicts(self.losses, {other.name: other})
        self.tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return self

    def __radd__(self, other: Any):
        """Addition operator for when forward addition returned `NotImplemented`.

        For example, doing something like `None + LossInfo()` will use __radd__,
        whereas doing `LossInfo() + None` will use __add__.
        """
        if other is None:
            return self
        elif other == 0:
            return self
        if isinstance(other, Tensor):
            # TODO: Other could be a loss tensor, maybe create a LossInfo object for it?
            pass
        return NotImplemented

    def __mul__(self, coefficient: Union[float,Tensor]) -> "LossInfo":
        """ Scale each loss tensor by `coefficient`.

        Returns
        -------
        LossInfo
            returns a scaled LossInfo instance.
        """
        # logger.debug(f" mul ({self.name}): before: {self.total_loss.requires_grad}")
        result = LossInfo(
            name=self.name,
            coefficient=self.coefficient * coefficient,
            total_loss=self.total_loss * coefficient,
            losses=OrderedDict([
                (k, value * coefficient) for k, value in self.losses.items()
            ]),
            metrics=self.metrics,
            tensors=self.tensors,
        )
        # logger.debug(f" mul ({self.name}): after: {result.total_loss.requires_grad}")
        return result

    def __truediv__(self, coefficient: Union[float, Tensor]) -> "LossInfo":
        # logger.debug(f" div ({self.name}): before: {self.total_loss.requires_grad}")
        
        # if coefficient != 0:
        result = self * (1 / coefficient)
        # logger.debug(f" div ({self.name}): after: {result.total_loss.requires_grad}")
        return result
        # else:
        #     raise RuntimeError("Can't divide stuff")
        # return NotImplemented

    @property
    def unscaled_losses(self):
        return OrderedDict([
            (k, value / self.coefficient) for k, value in self.losses.items()
        ])

    def to_log_dict(self, verbose: bool=False) -> Dict[str, Union[str, float, Dict]]:
        # TODO: Could also produce some wandb plots and stuff here
        return self.to_dict()

    def to_pbar_message(self):
        """ Smaller, less-detailed version of `self.to_log_dict()` (doesn't recurse into sublosses)
        meant to be used in progressbars.
        """
        message: Dict[str, Union[str, float]] = OrderedDict()
        message["Loss"] = float(self.total_loss.item())

        if self.metric:
            message[self.name] = self.metric.to_pbar_message()

        for name, loss_info in self.losses.items():
            message[name] = loss_info.to_pbar_message()

        message = add_prefix(message, prefix=self.name, sep=" ")

        return cleanup(message, sep=" ")
    
    def to_dict(self):
        self.drop_tensors()
        return super().to_dict()
    
    def drop_tensors(self) -> None:
        self.tensors.clear()
        for _, loss in self.losses.items():
            loss.drop_tensors()
        return self

    def detach(self) -> "LossInfo":
        """ 'detaches' this LossInfo object. 
        TODO: At the moment, does this by serializing and then deserializing,
        which works, but is a bit hacky. We could instead return a new LossInfo
        object where all the tensors and metrics and sublosses have also been detached.
        """ 
        return type(self).from_dict(self.to_dict())

    def absorb(self, other: "LossInfo") -> None:
        """Absorbs `other` into `self`, merging the losses and metrics.

        Args:
            other (LossInfo): Another loss to 'merge' into this one.
        """
        new_name = self.name
        old_name = other.name
        new_other = LossInfo(name=new_name)
        new_other.total_loss = other.total_loss
        # acumulate the metrics:
        new_other.metrics = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.metrics.items() 
        ])
        new_other.losses = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.losses.items() 
        ])
        self += new_other
    
    def all_metrics(self) -> Dict[str, Metrics]:
        """ Returns a 'cleaned up' dictionary of all the Metrics objects. """
        result: Dict[str, Metrics] = {}
        result.update(self.metrics)

        for name, loss in self.losses.items():
            result.update(loss.all_metrics())
        
        result = add_prefix(result, prefix=self.name, sep=" ")
        return result

    def get_supervised_metrics(self) -> Union[ClassificationMetrics, RegressionMetrics]:
        from tasks.tasks import Tasks

        if Tasks.SUPERVISED in self.metrics:
            return self.metrics[Tasks.SUPERVISED]
        all_metrics = self.all_metrics()
        if Tasks.SUPERVISED in all_metrics:
            return all_metrics[Tasks.SUPERVISED]
        
        suffix = Tasks.SUPERVISED
        # Return the first Metrics object which has a key ending with Tasks.SUPERVISED.
        for name, metric in all_metrics.items():
            if name.endswith(suffix):
                return metric

        raise RuntimeError(f"Couldn't find supervised metrics in lossinfo object {self}")

    def get_supervised_accuracy(self) -> float:
        return self.get_supervised_metrics().accuracy


@dataclass
class TrainValidLosses(Serializable):
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

    def items(self) -> Iterable[Tuple[int, Tuple[Optional[LossInfo], Optional[LossInfo]]]]:
        train_keys = set(self.train_losses).union(set(self.valid_losses))
        for k in sorted(train_keys):
            yield k, (self.train_losses.get(k), self.valid_losses.get(k))

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
