from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class Metrics:
    n_samples: int = 0
    accuracy: float = 0.
    
    @classmethod
    def from_tensors(cls, x: Tensor=None, h_x: Tensor=None, y_pred: Tensor=None, y: Tensor=None):
        # get the batch size:
        n_samples = 0
        for tensor in [x, h_x, y_pred, y]:
            if tensor is not None:
                n_samples = tensor.shape[0]

        acc = 0.
        #TODO: add other useful metrics.
        if y_pred is not None and y is not None:
            acc = accuracy(y_pred, y)
        return cls(n_samples=n_samples, accuracy=acc)

    def __add__(self, other: "Metrics") -> "Metrics":
        result = Metrics()
        self_total_acc = self.accuracy * self.n_samples
        other_total_acc = other.accuracy * other.n_samples
        result.n_samples = self.n_samples + other.n_samples
        if result.n_samples != 0:
            result.accuracy = (self_total_acc + other_total_acc) / result.n_samples
        return result

    def __repr__(self) -> str:
        return f"Metrics(n_samples={self.n_samples}, accuracy={self.accuracy:.2%})"

def accuracy(y_pred: Tensor, y: Tensor) -> float:
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=1)
    acc = (predicted == y).sum(dtype=float) / batch_size
    return acc.item()
