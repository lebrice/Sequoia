""" Module that defines a `Loss` class that holds losses and associated metrics.

This Loss object is used to bundle together the Loss and the Metrics. This is
in essence similar to the `TrainResult` and `EvalResult` dict objects from
pytorch_lightning.

Loss objects are used to simplify training with multiple "loss signals"
(e.g. in Self-Supervised Learning) by keeping track of the contribution of each
individual 'task' to the total loss, as well as their corresponding metrics.

For example:
>>> loss = Loss("total")
>>> loss += Loss("task_a", loss=1.23, metrics={"accuracy": 0.95})
>>> loss += Loss("task_b", loss=2.10)
>>> loss += Loss("task_c", loss=3.00)
>>> loss.to_log_dict()
{'loss': 6.33, 'losses/task_a/loss': 1.23, 'losses/task_a/accuracy': 0.95, \
'losses/task_b/loss': 2.1, 'losses/task_c/loss': 3.0}

Another cool feature of Loss objects is that they can automatically generate
relevant metrics when the relevant tensors are passed.


For example, consider a classification problem:
>>> y_pred = torch.eye(3)
>>> y_pred
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
>>> y = [0, 1, 1]
>>> loss = Loss("test", y_pred=y_pred, y=y)
>>> loss.metric
ClassificationMetrics(n_samples=3, accuracy=0.6666666865348816)

Or, consider a regression problem:
>>> y = [0.0, 1.0, 2.0, 3.0]
>>> y_pred = [0.0, 1.0, 2.0, 5.0] # mse = 1/4 * (5-3)**2 == 1.0
>>> reg_loss = Loss("test", y_pred=y_pred, y=y)
>>> reg_loss.metric
RegressionMetrics(n_samples=4, mse=tensor(1.))

See the `Loss` constructor for more info on which tensors are accepted.
"""
from collections import OrderedDict
from dataclasses import InitVar, dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from simple_parsing import field
from simple_parsing.helpers import dict_field
from utils.json_utils import Serializable, detach, move
from utils.logging_utils import cleanup, get_logger
from utils.utils import add_dicts, add_prefix

from .metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                      get_metrics)

logger = get_logger(__file__)

@dataclass
class Loss(Serializable):
    """ Object used to store the losses and metrics for a given 'learning task'. 
    
    Used to simplify the return type of the different `get_loss` functions and
    also to help in debugging models that a combination of different loss
    signals.

    TODO: Add some kind of histogram plot to show the relative contribution of
    each loss signal?
    TODO: Maybe create a `make_plots()` method to create wandb plots?
    """
    name: str
    loss: Tensor = 0.  # type: ignore
    losses: Dict[str, "Loss"] = dict_field()
    # NOTE: By setting to_dict=False below, we don't include the tensors when
    # serializing the attributes.
    # TODO: Does that also mean that the tensors can't be pickled (moved) by
    # pytorch-lightning during training? Is there a case where that would be
    # useful?
    tensors: Dict[str, Tensor] = dict_field(repr=False, to_dict=False)
    metrics: Dict[str, Metrics] = dict_field()
    # When multiplying the Loss by a value, this keep track of the coefficients
    # used, so that if we wanted to we could recover the 'unscaled' loss.
    _coefficient: Union[float, Tensor] = field(1.0, repr=False)

    x: InitVar[Optional[Tensor]] = None
    h_x: InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y: InitVar[Optional[Tensor]] = None

    def __post_init__(self,
                      x: Tensor = None,
                      h_x: Tensor = None,
                      y_pred: Tensor = None,
                      y: Tensor = None):
        assert self.name, "Loss objects should be given a name!"
        if self.name not in self.metrics:
            # Create a Metrics object if given the necessary tensors.
            metrics = get_metrics(x=x, h_x=h_x, y_pred=y_pred, y=y)
            if metrics:
                self.metrics[self.name] = metrics 

        for name, tensor in self.tensors.items():
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)

    @property
    def total_loss(self) -> Tensor:
        return self.loss

    @property
    def metric(self) -> Optional[Metrics]:
        """Shortcut for `self.metrics[self.name]`.

        Returns:
            Optional[Metrics]: The metrics associated with this Loss.
        """
        return self.metrics.get(self.name)

    @property
    def accuracy(self) -> float:
        if isinstance(self.metric, ClassificationMetrics):
            return self.metric.accuracy
        return self.get_supervised_accuracy()
    
    @property
    def mse(self) -> Tensor:
        assert isinstance(self.metric, RegressionMetrics), self
        return self.metric.mse

    def __add__(self, other: Any) -> "Loss":
        """Adds two Loss instances together.
        
        Adds the losses, total loss and metrics. Overwrites the tensors.
        Keeps the name of the first one. This is useful when doing something
        like:
        
        ```
        loss = Loss("Test")
        for x, y in dataloader:
            loss += model.get_loss(x=x, y=y)
        ```      
        
        Returns
        -------
        Loss
            The merged/summed up Loss.
        """
        if other == 0:
            return self
        if not isinstance(other, Loss):
            return NotImplemented
        name = self.name
        loss = self.loss + other.loss
        
        if self.name == other.name:
            losses  = add_dicts(self.losses, other.losses)
            metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire Loss
            # object into the 'losses' dict, rather than a single loss tensor.
            losses = add_dicts(self.losses, {other.name: other})
            # TODO: setting in the 'metrics' dict, we are duplicating the
            # metrics, since they now reside in the `self.metrics[other.name]`
            # and `self.losses[other.name].metrics` attributes.
            metrics = self.metrics
            # metrics = add_dicts(self.metrics, {other.name: other.metrics})
        
        tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return Loss(
            name=name,
            loss=loss,
            losses=losses,
            tensors=tensors,
            metrics=metrics,
            _coefficient=self._coefficient,
        )
    
    def __iadd__(self, other: "Loss") -> "Loss":
        """Adds Loss to `self` in-place.
        
        Adds the losses, total loss and metrics. Overwrites the tensors.
        Keeps the name of the first one. This is useful when doing something
        like:
        
        ```
        loss = Loss("Test")
        for x, y in dataloader:
            loss += model.get_loss(x=x, y=y)
        ```
        
        Returns
        -------
        Loss
            `self`: The merged/summed up Loss.
        """
        self.loss = self.loss + other.loss
        
        if self.name == other.name:
            self.losses  = add_dicts(self.losses, other.losses)
            self.metrics = add_dicts(self.metrics, other.metrics)
        else:
            # IDEA: when the names don't match, store the entire Loss
            # object into the 'losses' dict, rather than a single loss tensor.
            self.losses = add_dicts(self.losses, {other.name: other})
        
        self.tensors = add_dicts(self.tensors, other.tensors, add_values=False)
        return self

    def __radd__(self, other: Any):
        """Addition operator for when forward addition returned `NotImplemented`.

        For example, doing something like `None + Loss()` will use __radd__,
        whereas doing `Loss() + None` will use __add__.
        """
        if other is None:
            return self
        elif other == 0:
            return self
        if isinstance(other, Tensor):
            # TODO: Other could be a loss tensor, maybe create a Loss object for it?
            pass
        return NotImplemented

    def __mul__(self, factor: Union[float, Tensor]) -> "Loss":
        """ Scale each loss tensor by `coefficient`.

        Returns
        -------
        Loss
            returns a scaled Loss instance.
        """
        # logger.debug(f" mul ({self.name}): before: {self.loss.requires_grad}")
        result = Loss(
            name=self.name,
            loss=self.loss * factor,
            losses=OrderedDict([
                (k, value * factor) for k, value in self.losses.items()
            ]),
            metrics=self.metrics,
            tensors=self.tensors,
            _coefficient=self._coefficient * factor,
        )
        # logger.debug(f" mul ({self.name}): after: {result.loss.requires_grad}")
        return result

    def __truediv__(self, coefficient: Union[float, Tensor]) -> "Loss":
        # logger.debug(f" div ({self.name}): before: {self.loss.requires_grad}")
        
        # if coefficient != 0:
        result = self * (1 / coefficient)
        # logger.debug(f" div ({self.name}): after: {result.loss.requires_grad}")
        return result
        # else:
        #     raise RuntimeError("Can't divide stuff")
        # return NotImplemented

    @property
    def unscaled_losses(self):
        """ Recovers the 'unscaled' version of this loss.

        TODO: This isn't used anywhere. We could probably remove it.
        """
        return OrderedDict([
            (k, value / self._coefficient) for k, value in self.losses.items()
        ])

    def to_log_dict(self, verbose: bool = False) -> Dict[str, Union[str, float, Dict]]:
        """Creates a dictionary to be logged (e.g. by `wandb.log`).

        Args:
            verbose (bool, optional): Wether to include a lot of information, or
            to only log the 'essential' stuff. See the `cleanup` function for
            more info. Defaults to False.

        Returns:
            Dict: A dict containing the things to be logged.
        """
        # TODO: Could also produce some wandb plots and stuff here when verbose?
        log_dict = self.to_dict()
        keys_to_remove: List[str] = []
        if not verbose:
            # when NOT verbose, remove any entries with this matching key.
            # TODO: add/remove keys here if you want to customize what doesn't get logged to wandb.
            # TODO: Could maybe make this a class variable so that it could be
            # extended/overwritten, but that sounds like a bit too much rn.
            keys_to_remove = [
                "n_samples",
                "name",
                "confusion_matrix",
                "class_accuracy",
                "_coefficient",
            ]
        return cleanup(log_dict, keys_to_remove=keys_to_remove)

    def to_pbar_message(self):
        """ Smaller, less-detailed version of `self.to_log_dict()` (doesn't recurse into sublosses)
        meant to be used in progressbars.
        """
        message: Dict[str, Union[str, float]] = OrderedDict()
        message["Loss"] = float(self.loss)

        if self.metric:
            message[self.name] = self.metric.to_pbar_message()

        for name, loss_info in self.losses.items():
            message[name] = loss_info.to_pbar_message()

        message = add_prefix(message, prefix=self.name, sep=" ")

        return cleanup(message, sep=" ")

    def clear_tensors(self) -> None:
        """ Clears the `tensors` attribute of `self` and of sublosses.
        
        NOTE: This could be useful if you want to save some space/compute, but
        it isn't being used atm, and there's no issue. You might want to call
        this if you are storing big tensors (or passing them to the constructor)
        """
        self.tensors.clear()
        for _, loss in self.losses.items():
            loss.clear_tensors()
        return self

    def absorb(self, other: "Loss") -> None:
        """Absorbs `other` into `self`, merging the losses and metrics.

        Args:
            other (Loss): Another loss to 'merge' into this one.
        """
        new_name = self.name
        old_name = other.name
        # Here we create a new 'other' and use __iadd__ to merge the attributes.
        new_other = Loss(name=new_name)
        new_other.loss = other.loss
        # We also replace the name in the keys, if present.
        new_other.metrics = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.metrics.items() 
        ])
        new_other.losses = OrderedDict([
            (k.replace(old_name, new_name), v) for k, v in other.losses.items() 
        ])
        self += new_other

    def all_metrics(self) -> Dict[str, Metrics]:
        """ Returns a 'cleaned up' dictionary of all the Metrics objects. """
        assert self.name
        result: Dict[str, Metrics] = {}
        result.update(self.metrics)

        for name, loss in self.losses.items():
            # TODO: Aren't we potentially colliding with 'self.metrics' here?
            subloss_metrics = loss.all_metrics()
            for key, metric in subloss_metrics.items():
                assert key not in result, (
                    f"Collision in metric keys of subloss {name}: key={key}, "
                    f"result={result}"
                )
                result[key] = metric
        result = add_prefix(result, prefix=self.name, sep="/")
        return result


if __name__ == "__main__":
    import doctest
    doctest.testmod()
