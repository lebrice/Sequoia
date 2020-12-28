import gym
import numpy as np
from gym import Space, spaces
from gym.spaces import Box, Discrete
from gym.vector.utils import batch_space

from .named_tuple import NamedTuple, NamedTupleSpace


def test_basic():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2,2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2,2)),
    )
    v = named_tuple_space.sample()
    print(v)
    assert v in named_tuple_space
    # TODO: Maybe re-use all the tests for gym.spaces.Tuple in the gym repo
    # somehow?

    normal_tuple_space = spaces.Tuple([
        Box(0, 1, (2,2)),
        Discrete(2),
        Box(0, 1, (2,2)),
    ])
    assert normal_tuple_space.sample() in named_tuple_space
    assert named_tuple_space.sample() in normal_tuple_space


class StateTransition(NamedTuple):
    current_state: np.ndarray
    action: int
    next_state: np.ndarray


def test_basic_with_dtype():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2,2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2,2)),
        dtype=StateTransition,
    )
    v = named_tuple_space.sample()
    assert v in named_tuple_space
    assert isinstance(v, StateTransition)

    normal_tuple_space = spaces.Tuple([
        Box(0, 1, (2,2)),
        Discrete(2),
        Box(0, 1, (2,2)),
    ])
    assert normal_tuple_space.sample() in named_tuple_space
    assert named_tuple_space.sample() in normal_tuple_space


def test_isinstance_namedtuple():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2,2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2,2)),
        dtype=StateTransition,
    )
    assert isinstance(named_tuple_space, NamedTupleSpace)
    assert isinstance(named_tuple_space.sample(), NamedTuple)



def test_equals_tuple_space_with_same_items():
    """ Test that a NamedTupleSpace is considered equal to a Tuple space if
    the spaces are in the same order and all equal (regardless of the names).
    """
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2,2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2,2)),
        dtype=StateTransition,
    )
    tuple_space = spaces.Tuple([
        Box(0, 1, (2,2)),
        Discrete(2),
        Box(0, 1, (2,2)),
    ])
    assert named_tuple_space == tuple_space
    assert tuple_space == named_tuple_space


def test_batch_space_is_singledispatch():
    import gym
    from gym.vector.utils import batch_space

    from .named_tuple import NamedTupleSpace
    assert hasattr(gym.vector.utils.batch_space, "registry")


def test_batch_space():
    named_tuple_space = NamedTupleSpace(
        current_state=Box(0, 1, (2,2)),
        action=Discrete(2),
        next_state=Box(0, 1, (2,2)),
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
