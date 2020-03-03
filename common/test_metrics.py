from .metrics import Metrics, accuracy, per_class_accuracy, confusion_matrix
from .losses import LossInfo
import torch
import numpy as np


def test_metrics_add_properly():        
    m1 = Metrics(n_samples=10, accuracy=0.8)
    m2 = Metrics(n_samples=1, accuracy=1)
    m3 = m1 + m2
    assert m3 == Metrics(n_samples=11, accuracy=9/11)
    m1 += m2
    assert m1 == Metrics(n_samples=11, accuracy=9/11)

def test_metrics_from_tensors():
    y_pred = torch.Tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.Tensor([
        1,
        2,
        0,
    ])
    m = Metrics.from_tensors(y_pred=y_pred, y=y)
    assert m.n_samples == 3
    assert m.accuracy == 2/3

def test_accuracy():
    y_pred = torch.Tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.Tensor([
        1,
        2,
        0,
    ])
    assert accuracy(y_pred, y) == 2/3


def test_per_class_accuracy_perfect():
    y_pred = torch.Tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.0, 0.9],
        [0.1, 0.4, 0.5],
        [0.9, 0.1, 0.0],
    ])
    y = torch.Tensor([
        1,
        2,
        2,
        0,
    ])
    expected = [1, 1, 1]
    class_acc = per_class_accuracy(y_pred, y).numpy().tolist()
    assert class_acc == expected


def test_per_class_accuracy_zero():
    y_pred = torch.Tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
        [0.1, 0.9, 0.0],
    ])
    y = torch.Tensor([
        0,
        0,
        0,
        0,
    ])
    expected = [0, 0, 0]
    class_acc = per_class_accuracy(y_pred, y).numpy().tolist()
    assert class_acc == expected


def test_confusion_matrix():
    y_pred = torch.Tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.4, 0.5],
        [0.1, 0.9, 0.0],
        [0.9, 0.0, 0.1],
    ])
    y = torch.Tensor([
        0,
        0,
        1,
        0,
    ])
    expected = [
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
    ]
    confusion_mat = confusion_matrix(y_pred, y).numpy().tolist()
    assert confusion_mat == expected

def test_per_class_accuracy_realistic():
    y_pred = torch.Tensor([
        [0.9, 0.0, 0.0], # correct for class 0
        [0.1, 0.5, 0.4], # correct for class 1
        [0.1, 0.0, 0.9], # correct for class 2
        [0.1, 0.8, 0.1], # wrong, should be 1
        [0.1, 0.0, 0.9], # wrong, should be 0
        [0.9, 0.0, 0.0], # wrong, should be 1
        [0.1, 0.5, 0.4], # wrong, should be 2
        [0.1, 0.4, 0.5], # correct for class 2
    ])
    y = torch.Tensor([
        0,
        1,
        2,
        0, 
        0,
        1,
        2,
        2,
    ])
    expected = [1/3, 1/2, 2/3]
    class_acc = per_class_accuracy(y_pred, y).numpy().tolist()
    assert all(np.isclose(class_acc, expected))