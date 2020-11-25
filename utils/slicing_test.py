from .slicing import get_slice, set_slice
import pytest
import numpy as np
import torch


@pytest.mark.parametrize("source, indices, target",
[
    (np.arange(10), np.arange(5), np.arange(5)),
    ({"a": np.arange(10), "b": np.arange(10)}, np.arange(5), {"a": np.arange(5), "b": np.arange(5)}),
    (({"a": np.arange(10)}, np.arange(10) + 5), 3, ({"a": 3}, 8)),
])
def test_get_slice(source, indices, target):
    assert str(get_slice(source, indices)) == str(target)


@pytest.mark.parametrize("target, indices, values, result",
[
    (
        np.arange(10, dtype=float),
        np.arange(5),
        np.zeros(5),
        np.concatenate([np.zeros(5), np.arange(5) + 5.])),
    (
        {"a": np.arange(10, dtype=float), "b": np.zeros(10)},
        np.arange(10),
        {"a": np.ones(10), "b": np.ones(10)},
        {"a": np.ones(10), "b": np.ones(10)},
    ),
    (
        ({"a": np.arange(10)}, np.arange(10) + 5),
        0,
        ({"a": 3}, 8),
        (
            {"a": np.concatenate([np.array([3]), 1  + np.arange(9)])},
            np.concatenate([np.array([8]), 6 + np.arange(9)]),
        ),
    ),
])
def test_set_slice(target, indices, values, result):
    set_slice(target, indices, values)
    assert str(target) == str(result)
