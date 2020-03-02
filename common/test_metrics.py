from .metrics import Metrics, accuracy
from .losses import LossInfo
import torch

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