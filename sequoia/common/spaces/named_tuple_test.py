import numpy as np
import pytest
from gym import spaces
from gym.spaces import Box, Discrete
from gym.vector.utils import batch_space

from .named_tuple import NamedTuple, NamedTupleSpace

pytestmark = pytest.mark.skip(
    reason="Removing the NamedTuple space and NamedTuple class in favour of TypedDict.",
)


def test_basic():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
    )
    v = named_tuple_space.sample()
    print(v)
    assert v in named_tuple_space
    # TODO: Maybe re-use all the tests for gym.spaces.Tuple in the gym repo
    # somehow?

    normal_tuple_space = spaces.Tuple(
        [
            Box(0, 1, (2, 2)),
            Discrete(2),
            Box(0, 1, (2, 2)),
        ]
    )
    assert normal_tuple_space.sample() in named_tuple_space
    assert named_tuple_space.sample() in normal_tuple_space


class StateTransition(NamedTuple):
    current_state: np.ndarray
    action: int
    next_state: np.ndarray


def test_basic_with_dtype():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    v = named_tuple_space.sample()
    assert v in named_tuple_space
    assert isinstance(v, StateTransition)

    normal_tuple_space = spaces.Tuple(
        [
            Box(0, 1, (2, 2)),
            Discrete(2),
            Box(0, 1, (2, 2)),
        ]
    )
    assert normal_tuple_space.sample() in named_tuple_space
    assert named_tuple_space.sample() in normal_tuple_space


@pytest.mark.xfail()
def test_isinstance_namedtuple():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    assert isinstance(named_tuple_space, NamedTupleSpace)
    assert isinstance(named_tuple_space.sample(), NamedTuple)


def test_equals_tuple_space_with_same_items():
    """Test that a NamedTupleSpace is considered equal to a Tuple space if
    the spaces are in the same order and all equal (regardless of the names).
    """
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    tuple_space = spaces.Tuple(
        [
            Box(0, 1, (2, 2)),
            Discrete(2),
            Box(0, 1, (2, 2)),
        ]
    )
    assert named_tuple_space == tuple_space
    assert tuple_space == named_tuple_space


def test_batch_objets_considered_valid_samples():
    from dataclasses import dataclass

    import numpy as np

    from sequoia.common.batch import Batch

    @dataclass(frozen=True)
    class StateTransitionDataclass(Batch):
        current_state: np.ndarray
        action: int
        next_state: np.ndarray

    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransitionDataclass,
    )
    obs = StateTransitionDataclass(
        current_state=np.ones([2, 2]) / 2,
        action=1,
        next_state=np.zeros([2, 2]),
    )
    assert obs in named_tuple_space
    assert named_tuple_space.sample() in named_tuple_space
    assert isinstance(named_tuple_space.sample(), StateTransitionDataclass)


def test_batch_space():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2, 2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2, 2)),
        dtype=StateTransition,
    )
    assert batch_space(named_tuple_space, n=5) == NamedTupleSpace(
        current_state=Box(0, 1, (5, 2, 2)),
        action=spaces.MultiDiscrete([2, 2, 2, 2, 2]),
        next_state=Box(0, 1, (5, 2, 2)),
        dtype=StateTransition,
    )


## IDEA: Creating a space like this, using the same syntax as with NamedTuple
# class StateTransitionSpace(NamedTupleSpace):
#     current_state: Box = Box(0, 1, (2,2))
#     action: Discrete = Discrete(2)
#     current_state: Box = Box(0, 1, (2,2))

# space = StateTransitionSpace()
# space.sample()
