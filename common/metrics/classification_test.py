import numpy as np
import torch

from .classification import ClassificationMetrics
from .get_metrics import get_metrics

def test_classification_metrics_add_properly():        
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        0,
    ])
    m1 = ClassificationMetrics(y_pred=y_pred, y=y)
    assert m1.n_samples == 3
    assert np.isclose(m1.accuracy, 2/3)
    
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        2,
        0,
        0,
    ])
    m2 = ClassificationMetrics(y_pred=y_pred, y=y)
    assert m2.n_samples == 5
    assert np.isclose(m2.accuracy, 3/5)
    assert all(np.isclose(m2.class_accuracy, [0, 1, 1]))

    m3 = m1 + m2
    assert m3.n_samples == 8
    assert np.isclose(m3.accuracy, 5/8)


def test_metrics_from_tensors():
    y_pred = torch.as_tensor([
        [0.01, 0.90, 0.09],
        [0.01, 0, 0.99],
        [0.01, 0, 0.99],
    ])
    y = torch.as_tensor([
        1,
        2,
        0,
    ])
    m = get_metrics(y_pred=y_pred, y=y)
    assert m.n_samples == 3
    assert np.isclose(m.accuracy, 2/3)
