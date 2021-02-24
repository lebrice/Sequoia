""" Utility functions for calculating metrics. """
import torch
from torch import Tensor
from typing import Union, Optional
import numpy as np
import functools


@torch.no_grad()
def get_confusion_matrix(y_pred: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor], num_classes: int = None) -> Union[Tensor, np.ndarray]:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348

    NOTE: `y_pred` is assumed to be the logits with shape [B, C], while the
    labels `y` is assumed to have shape either `[B]` or `[B, 1]`, unless `num_classes`
    is given, in which case y_pred can be the predicted labels.
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()

    # FIXME: How do we properly check if something is an integer type in np?
    if len(y_pred.shape) == 1 and y_pred.dtype not in {np.float32, np.float64}:
        # y_pred is already the predicted labels.
        y_preds = y_pred
        if num_classes is None:
            raise NotImplementedError(f"Can't determine the number of classes. Pass logits rather than predicted labels.")
        n_classes = num_classes
    elif y_pred.shape[-1] == 1:
        n_classes = 2  # y_pred is the logit for binary classification.
        y_preds = y_pred.round()
    else:
        # y_pred is assumed to be the logits.
        n_classes = y_pred.shape[-1]
        y_preds = y_pred.argmax(-1)

    y = y.flatten().astype(int)
    y_preds = y_preds.flatten().astype(int)

    # BUG: This is failing on the last batch.
    assert y.shape == y_preds.shape, (y.shape, y_preds.shape)
    assert y.dtype == y_preds.dtype == np.int, (y.dtype, y_preds.dtype)

    confusion_matrix = np.zeros([n_classes, n_classes])
    
    assert 0 <= y.min() and y.max() < n_classes, (y, n_classes)
    assert 0 <= y_preds.min() and y_preds.max() < n_classes, (y_preds, n_classes)
    
    for y_t, y_p in zip(y, y_preds):
        confusion_matrix[y_t, y_p] += 1
    return confusion_matrix

@torch.no_grad()
def accuracy(y_pred: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> float:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(-1)
    acc = (predicted == y).sum(dtype=float) / batch_size
    return acc.item()

@torch.no_grad()
def get_accuracy(confusion_matrix: Union[Tensor, np.ndarray]) -> float:
    if isinstance(confusion_matrix, Tensor):
        diagonal = confusion_matrix.diag()
    else:
        diagonal = np.diag(confusion_matrix)
    return (diagonal.sum() / confusion_matrix.sum()).item()

@torch.no_grad()
def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    return get_class_accuracy(confusion_mat)

@torch.no_grad()
def get_class_accuracy(confusion_matrix: Tensor) -> Tensor:
    if isinstance(confusion_matrix, Tensor):
        diagonal = confusion_matrix.diag()
    else:
        diagonal = np.diag(confusion_matrix)
    sum_of_columns = confusion_matrix.sum(1)
    if isinstance(confusion_matrix, Tensor):
        sum_of_columns.clamp_(min=1e-10)
    else:
        sum_of_columns = sum_of_columns.clip(min=1e-10)
    return diagonal / sum_of_columns
