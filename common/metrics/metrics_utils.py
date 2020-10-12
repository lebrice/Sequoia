""" Utility functions for calculating metrics. """
import torch
from torch import Tensor
from typing import Union, Optional
import numpy as np
import functools


def convert_to_and_back(numpy: bool = True,
                        tensor: Optional[bool] = None,
                        device: Optional[torch.device] = None):
    """ Creates wrapper that converts args to ndarrays or tensors and then converts
    back the results of the function to the original type.
    
    If `numpy` is True (default), converts function arguments that are tensors
    to ndarrays and converts results back to tensors.
    Otherwise, if `tensor` is True, converts function arguments that are numpy
    arrays to tensors, and converts results back to numpy arrays.
    Optionally, when converting ndarrays to tensors, `device` can be passed.
    When `numpy` is True, the default device for the function outputs will be
    the device of the first Tensor encountered.
    When `tensor` is True, the default device will be "cuda" if available, else
    "cpu".
    """
    # TODO: convert tensor args to numpy arrays before passing them to the
    # function, and then convert the results back to Tensor. 
    if tensor is not None:
        numpy = not tensor
    else:
        tensor = not numpy
    assert numpy ^ tensor, f"Only one of `numpy` or `tensor` can be set."
    if tensor and device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal device
            # Wether inputs were tensors, in which case this will return a tensor.
            # A tensor to grab the device/dtype from later.
            new_args = []
            for arg in args:
                if isinstance(arg, Tensor) and numpy:
                    device = arg.device if device is None else device
                    arg = arg.detach().cpu().numpy()
                elif isinstance(arg, np.ndarray) and tensor:
                    arg = torch.as_tensor(arg, device=device)
                new_args.append(arg)
            new_kwargs = {}
            for name, val in kwargs.items():
                if isinstance(val, Tensor) and numpy:
                    device = val.device if device is None else device
                    val = val.detach().cpu().numpy()
                elif isinstance(arg, np.ndarray) and tensor:
                    val = torch.as_tensor(arg, device=device)
                new_kwargs[name] = val

            outputs = fn(*new_args, **new_kwargs)

            if isinstance(outputs, np.ndarray) and numpy and device is not None:
                return torch.as_tensor(outputs, device=device)
            if isinstance(outputs, Tensor) and tensor:
                return outputs.detach().cpu().numpy()
            return outputs

        return wrapped
    return wrapper

# @convert_to_and_back(numpy=True)
@torch.no_grad()
def get_confusion_matrix(y_pred: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> Union[Tensor, np.ndarray]:
    """ Taken from https://discuss.pytorch.org/t/how-to-find-individual-class-accuracy/6348
    
    NOTE: `y_pred` is assumed to be the logits with shape [B, C], while the
    labels `y` is assumed to have shape either `[B]` or `[B, 1]`.
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()
    
    # FIXME: How do we properly check if something is an integer type in np?
    if len(y_pred.shape) == 1 and y_pred.dtype not in {np.float32, np.float64}:
        # y_pred is already the predicted labels.
        y_preds = y_pred
        raise NotImplementedError(f"Can't determine the number of classes. Pass logits rather than predicted labels.")
    elif y_pred.shape[-1] == 1:
        n_classes = 2  # y_pred is the logit for binary classification.
        y_preds = y_pred.round()
    else:
        n_classes = y_pred.shape[-1]
        y_preds = y_pred.argmax(-1)

    y = y.flatten().astype(int)
    y_preds = y_preds.flatten().astype(int)
    
    assert y.shape == y_preds.shape, (y.shape, y_preds.shape)
    assert y.dtype == y_preds.dtype == np.int, (y.dtype, y_preds.dtype)
    
    confusion_matrix = np.zeros([n_classes, n_classes])
    
    assert 0 <= y.min() and y.max() < n_classes, (y, n_classes)
    assert 0 <= y_preds.min() and y_preds.max() < n_classes, (y_preds, n_classes)
    
    for y_t, y_p in zip(y, y_preds):
        confusion_matrix[y_t, y_p] += 1
    return confusion_matrix

@convert_to_and_back(numpy=True)
@torch.no_grad()
def accuracy(y_pred: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]) -> float:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    batch_size = y_pred.shape[0]
    _, predicted = y_pred.max(dim=-1)
    acc = (predicted == y).sum(dtype=torch.float) / batch_size
    return acc.item()

@torch.no_grad()
def get_accuracy(confusion_matrix: Union[Tensor, np.ndarray]) -> float:
    if isinstance(confusion_matrix, Tensor):
        diagonal = confusion_matrix.diag()
    else:
        diagonal = np.diag(confusion_matrix)
    return (diagonal.sum() / confusion_matrix.sum()).item()

@convert_to_and_back(tensor=True)
@torch.no_grad()
def class_accuracy(y_pred: Tensor, y: Tensor) -> Tensor:
    confusion_mat = get_confusion_matrix(y_pred=y_pred, y=y)
    return get_class_accuracy(confusion_mat)

@convert_to_and_back(tensor=True)
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
