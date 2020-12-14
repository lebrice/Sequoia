from .slicing import get_slice, set_slice
import pytest
import numpy as np
import torch


from typing import NamedTuple, Type


class DummyTuple(NamedTuple):
    a: np.ndarray
    b: np.ndarray


@pytest.mark.parametrize("source, indices, expected",
[
    (np.arange(10), np.arange(5), np.arange(5)),
    ({"a": np.arange(10), "b": np.arange(10)}, np.arange(5), {"a": np.arange(5), "b": np.arange(5)}),
    (({"a": np.arange(10)}, np.arange(10) + 5), 3, ({"a": 3}, 8)),
    ( # Test with namedtuples.
        {
            "a": np.array([0, 1, 2]),
            "b": DummyTuple(a=np.zeros([3, 4]), b=np.ones([5, 4])),
        },
        np.arange(2),
        {
            "a": np.array([0, 1]),
            "b": DummyTuple(a=np.zeros([2, 4]), b=np.ones([2, 4]))
        }
    )
])
def test_get_slice(source, indices, expected):
    assert str(get_slice(source, indices)) == str(expected)


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
    ( # Test with NamedTuples.
        {
            "a": np.array([0, 1, 2]),
            "b": DummyTuple(a=np.zeros(5), b=np.ones(5)),
        },
        np.arange(2),
        {
            "a": np.array([5, 7]),
            "b": DummyTuple(a=np.ones(2), b=np.zeros(2))
        },
        {
            "a": np.array([5, 7, 2]),
            "b": DummyTuple(a=np.array([1., 1., 0., 0., 0.]), b=np.array([0., 0., 1., 1., 1.])),
        },
    )
])
def test_set_slice(target, indices, values, result):
    set_slice(target, indices, values)
    assert str(target) == str(result)


from .slicing import concatenate



@pytest.mark.parametrize("a, b, kwargs, expected",
[
    (np.array([0, 1, 2]), np.array([3, 4, 5, 6]), {}, np.arange(7)),
    (
        {
            "a": np.array([0, 1, 2]),
            "b": DummyTuple(a=np.zeros(3), b=np.ones(3)),
        },
        {
            "a": np.array([3, 4, 5]),
            "b": DummyTuple(a=np.zeros(4), b=np.ones(4)),
        },
        {},
        {
            "a": np.array([0, 1, 2, 3, 4, 5]),
            "b": DummyTuple(a=np.zeros(7), b=np.ones(7)),
        },
    ),
    (
        {
            "a": np.array([[0], [1], [2]]), # [3, 1]
            "b": DummyTuple(a=np.zeros([1, 4]), b=np.ones([1, 4])),
        },
        {
            "a": np.array([[3], [4], [5], [6]]), # shape [4, 1]
            "b": DummyTuple(a=np.zeros([2, 4]), b=np.ones([3, 4])),
        },
        {"axis": 0},
        {
            "a": np.array([[0], [1], [2], [3], [4], [5], [6]]),
            "b": DummyTuple(a=np.zeros([3, 4]), b=np.ones([4, 4])),
        },
    ),
])
def test_concat(a, b, kwargs, expected):
    assert str(concatenate(a, b, **kwargs)) == str(expected)
