from dataclasses import dataclass, field, InitVar
from typing import Dict, Optional
import torch
from torch import Tensor

@dataclass
class Metrics:
    n_samples: int = 0

    confusion_matrix: Optional[Tensor] = field(default=None, repr=False)

    # fields we generate from the confusion matrix (if provided)
    accuracy: float = field(default=0., init=False)
    class_accuracy: Optional[Tensor] = field(default=None, init=False)


    x:      InitVar[Optional[Tensor]] = None
    h_x:    InitVar[Optional[Tensor]] = None
    y_pred: InitVar[Optional[Tensor]] = None
    y:      InitVar[Optional[Tensor]] = None

    def __post_init__(self, x: Tensor=None, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None):
        # get the batch size:
        for tensor in [x, h_x, y_pred, y]:
            if tensor is not None:
                self.n_samples = tensor.shape[0]
                break
        if self.confusion_matrix is None and y_pred is not None and y is not None:
            self.confusion_matrix = get_confusion_matrix(y_pred=y_pred, y=y)

        #TODO: add other useful metrics (potentially ones using x or h_x?)
        if self.confusion_matrix is not None:
            self.accuracy = get_accuracy(self.confusion_matrix)
            self.class_accuracy = get_class_accuracy(self.confusion_matrix)

    def __add__(self, other: "Metrics") -> "Metrics":
        confusion_matrix: Optional[Tensor] = None
        if self.confusion_matrix is None:
            if other.confusion_matrix is None:
                confusion_matrix = None
            else:
                confusion_matrix = other.confusion_matrix.clone()
        elif other.confusion_matrix is None:
            confusion_matrix = self.confusion_matrix.clone()
        else:
            confusion_matrix = self.confusion_matrix + other.confusion_matrix

        result = Metrics(
            n_samples=self.n_samples + other.n_samples,
            confusion_matrix=confusion_matrix,
        )
        return result
    
    def to_log_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "class_accuracy": self.class_accuracy,
        }
    
    def __str__(self) -> str:
        return f"Metrics(n_samples={self.n_samples}, accuracy={self.accuracy:.2%})"


def get_confusion_matrix(y_pred: Tensor, y: Tensor) -> Tensor:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348
    """
    n_classes = y_pred.shape[-1]
    confusion_matrix = torch.zeros(n_classes, n_classes)
    _, y_preds = torch.max(y_pred, 1)
    for y_t, y_p in zip(y.view(-1), y_preds.view(-1)):
        confusion_matrix[y_t.long(), y_p.long()] += 1
    return confusion_matrix


def accuracy(y_pred: Tensor, y: Tensor) -> float:
    confusion_mat = get_confusion_matrix(y_pred, y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=1)
    acc = (predicted == y).sum(dtype=torch.float) / batch_size
    return acc.item()


def get_accuracy(confusion_matrix: Tensor) -> float:
    return (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()


def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred, y)
    return confusion_mat.diag()/confusion_mat.sum(1).clamp_(1e-10, 1e10)

def get_class_accuracy(confusion_matrix: Tensor) -> Tensor:
    return confusion_matrix.diag()/confusion_matrix.sum(1).clamp_(1e-10, 1e10)
