""" Metrics class for classification.

Gives the accuracy, the class accuracy, and the confusion matrix for a given set
of (raw/pre-activation) logits Tensor `y_pred` and the class labels `y`. 
"""
from dataclasses import dataclass, InitVar
from functools import total_ordering
from typing import Dict, Optional, Union, Any

import numpy as np
import torch
import wandb
from torch import Tensor

from simple_parsing import field
from sequoia.utils.serialization import detach, move

from .metrics import Metrics
from .metrics_utils import (get_accuracy, get_class_accuracy,
                            get_confusion_matrix)

# TODO: Might be a good idea to add a `task` attribute to Metrics or
# Loss objects, in order to check that we aren't adding the class
# accuracies or confusion matrices from different tasks by accident.
# We could also maybe add them but fuse them properly, for instance by
# merging the class accuracies and confusion matrices?
# 
# For example, if a first metric has class accuracy [0.1, 0.5] 
# (n_samples=100) and from a task with classes [0, 1] is added to a
# second Metrics with class accuracy [0.9, 0.8] (n_samples=100) for task
# with classes [0,3], the resulting Metrics object would have a 
# class_accuracy of [0.5 (from (0.1+0.9)/2 = 0.5), 0.5, 0 (no data), 0.8]
# n_samples would then also have to be split on a per-class basis.
# n_samples could maybe be just the sum of the confusion matrix entries?
# 
# As for the confusion matrices, they could be first expanded to fit the
# range of both by adding empty columns/rows to each and then be added
# together.


@dataclass
class ClassificationMetrics(Metrics):
    # fields we generate from the confusion matrix (if provided) or from the
    # forward pass tensors.
    accuracy: float = 0.
    confusion_matrix: Optional[Union[Tensor, np.ndarray]] = field(default=None, repr=False, compare=False)
    class_accuracy: Optional[Union[Tensor, np.ndarray]] = field(default=None, repr=False, compare=False)

    # Optional arguments used to create the attributes of the metrics above.
    # NOTE: These wont become attributes on the object, just args to postinit.
    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None
    num_classes: InitVar[Optional[int]] = None
    
    def __post_init__(self,
                      x: Tensor = None,
                      h_x: Tensor = None,
                      y_pred: Tensor = None,
                      y: Tensor = None,
                      num_classes: int = None):

        super().__post_init__(x=x, h_x=h_x, y_pred=y_pred, y=y)

        if self.confusion_matrix is None and y_pred is not None and y is not None:
            self.confusion_matrix = get_confusion_matrix(y_pred=y_pred, y=y, num_classes=num_classes)

        #TODO: add other useful metrics (potentially ones using x or h_x?)
        if self.confusion_matrix is not None:
            self.accuracy = get_accuracy(self.confusion_matrix)
            self.accuracy = round(self.accuracy, 6)
            self.class_accuracy = get_class_accuracy(self.confusion_matrix)


    def __add__(self, other: "ClassificationMetrics") -> "ClassificationMetrics":
        confusion_matrix: Optional[Tensor] = None
        if self.n_samples == 0:
            return other
        if not isinstance(other, ClassificationMetrics):
            return NotImplemented
        
        # Create the 'sum' confusion matrix:
        confusion_matrix: Optional[np.ndarray] = None
        if self.confusion_matrix is None and other.confusion_matrix is not None:
            confusion_matrix = other.confusion_matrix.clone()
        elif other.confusion_matrix is None:
            confusion_matrix = self.confusion_matrix.clone()
        else:
            confusion_matrix = self.confusion_matrix + other.confusion_matrix
        
        result = ClassificationMetrics(
            n_samples=self.n_samples + other.n_samples,
            confusion_matrix=confusion_matrix,
            num_classes=self.num_classes,
        )
        return result

    def to_log_dict(self, verbose=False):
        log_dict = super().to_log_dict(verbose=verbose)
        log_dict["accuracy"] = self.accuracy
        if verbose:
            # Maybe add those as plots, rather than tensors?
            log_dict["class_accuracy"] = self.class_accuracy
            log_dict["confusion_matrix"] = self.confusion_matrix
        return log_dict
    
    # def __str__(self):
    #     s = super().__str__()
    #     s = s.replace(f"accuracy={self.accuracy}", f"accuracy={self.accuracy:.3%}")
    #     return s

    def to_pbar_message(self) -> Dict[str, Union[str, float]]:
        message = super().to_pbar_message()
        message["acc"] = float(self.accuracy)
        return message

    def detach(self) -> "ClassificationMetrics":
        return ClassificationMetrics(
            n_samples=detach(self.n_samples),
            accuracy=float(self.accuracy),
            class_accuracy=detach(self.class_accuracy),
            confusion_matrix=detach(self.confusion_matrix),
        )

    def to(self, device: Union[str, torch.device]) -> "ClassificationMetrics":
        """Returns a new Metrics with all the attributes 'moved' to `device`."""
        return ClassificationMetrics(
            n_samples=move(self.n_samples, device),
            accuracy=move(self.accuracy, device),
            class_accuracy=move(self.class_accuracy, device),
            confusion_matrix=move(self.confusion_matrix, device),
        )

    @property
    def objective(self) -> float:
        return float(self.accuracy)

    # def __lt__(self, other: Union["ClassificationMetrics", Any]) -> bool:
    #     if isinstance(other, ClassificationMetrics):
    #         return self.accuracy < other.accuracy
    #     return NotImplemented

    # def __ge__(self, other: Union["ClassificationMetrics", Any]) -> bool:
    #     if isinstance(other, ClassificationMetrics):
    #         return self.accuracy >= other.accuracy
    #     return NotImplemented

    # def __eq__(self, other: Union["ClassificationMetrics", Any]) -> bool:
    #     if isinstance(other, ClassificationMetrics):
    #         return self.accuracy == other.accuracy and self.n_samples == other.n_samples
    #     return NotImplemented
