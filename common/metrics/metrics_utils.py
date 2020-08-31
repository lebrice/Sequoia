""" Utility functions for calculating metrics. """
import torch
from torch import Tensor


@torch.no_grad()
def get_confusion_matrix(y_pred: Tensor, y: Tensor) -> Tensor:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348
    """
    n_classes = y_pred.shape[-1]
    confusion_matrix = torch.zeros(n_classes, n_classes)
    y_pred = y_pred.reshape([-1, n_classes])
    _, y_preds = torch.max(y_pred, 1)
    for y_t, y_p in zip(y.view(-1), y_preds.view(-1)):
        confusion_matrix[y_t.long(), y_p.long()] += 1
    # print(y_pred, y, confusion_matrix)
    return confusion_matrix

@torch.no_grad()
def accuracy(y_pred: Tensor, y: Tensor) -> float:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=1)
    acc = (predicted == y).sum(dtype=torch.float) / batch_size
    return acc.item()


def get_accuracy(confusion_matrix: Tensor) -> float:
    return (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()


def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    return get_class_accuracy(confusion_mat)


def get_class_accuracy(confusion_matrix: Tensor) -> Tensor:
    return confusion_matrix.diag()/confusion_matrix.sum(1).clamp_(1e-10, 1e10)
