""" Utility functions for calculating metrics. """
import torch
from torch import Tensor
from typing import Union
import numpy as np


@torch.no_grad()
def get_confusion_matrix(y_pred: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> Tensor:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348
    
    NOTE: `y_pred` is assumed to be the logits with shape [B, C], while the
    labels `y` is assumed to have shape either `[B]` or `[B, 1]`.
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()
    assert len(y_pred.shape) == 2
    
    if y_pred.shape[-1] == 1:
        n_classes = 2  # y_pred is the logit for binary classification.
        y_preds = y_pred.round()
    else:
        n_classes = y_pred.shape[-1]
        y_preds = y_pred.argmax(-1)
    
    y = y.flatten().astype(int)
    y_preds = y_preds.flatten().astype(int)
    
    assert y.shape == y_preds.shape and y.dtype == y_preds.dtype == np.int
    confusion_matrix = np.zeros([n_classes, n_classes])
    assert 0 <= y.min() and y.max() < n_classes, (y, n_classes)
    assert 0 <= y_preds.min() and y_preds.max() < n_classes, (y_preds, n_classes)
    for y_t, y_p in zip(y, y_preds):
        confusion_matrix[y_t, y_p] += 1
    return confusion_matrix

@torch.no_grad()
def accuracy(y_pred: Tensor, y: Tensor) -> float:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=-1)
    acc = (predicted == y).sum(dtype=torch.float) / batch_size
    return acc.item()


@torch.no_grad()
def get_accuracy(confusion_matrix: Tensor) -> float:
    return (confusion_matrix.diag().sum() / confusion_matrix.sum().float()).item()


@torch.no_grad()
def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    return get_class_accuracy(confusion_mat)


def get_class_accuracy(confusion_matrix: Tensor) -> Tensor:
    return confusion_matrix.diag() / confusion_matrix.sum(1).float().clamp_(1e-10, 1e10)
