# TODO: Need tests for putting fancier objects in the ReplayBuffer.
import pytest
from .replay_buffer import ReplayBuffer
from sequoia.common.typed_gym import _Space
from typing import TypeVar

T = TypeVar("T")


@pytest.mark.xfail(reason="WIP")
def test_sampling_is_correct():
    """
    TODO: test that sampling from the buffer does indeed give a uniform representation of all the
    data seen so far (even when adding multiple batches).
    
    BUG: There is probably a bug in the replay buffer implementation, where the 'counter' used in
    the probability of adding a value is only local to a single count to `append`. Should instead
    be somekind of `n_seen_so_far` like in the Replay Buffer of @pclucas.
    """

from gym import spaces


@pytest.mark.xfail(reason="WIP")
@pytest.mark.parametrize("item_space", [
    spaces.Discrete(10),
    spaces.Box(0, 1, (3, 3)),
    spaces.Tuple([spaces.Discrete(10), spaces.Box(0, 1, (3, 3))]),
    spaces.Dict(a=spaces.Discrete(10), b=spaces.Box(0, 1, (3, 3))),
])
def test_replay_buffer(item_space: _Space[T]):
    # TODO: First, add tests for the env dataset / dataloader / experience replay with envs that
    # have typed objects (e.g.) Observation/Action/Reward, tensors, etc.
    capacity = 10

    buffer = ReplayBuffer[T](item_space=item_space, capacity=capacity, seed=123)
    assert len(buffer) == 0

    item = item_space.sample()
    buffer.append(item)
    assert len(buffer) == 1
    assert buffer[0] is item
    assert item in buffer

    assert buffer.sample(1) is item
    
    other_items = [item_space.sample() for _ in range(1, capacity)]
    buffer.add_reservoir(other_items)
    assert len(buffer) == capacity
    assert buffer.full

    for i, item in zip(range(1, capacity), other_items):
        assert buffer[i] == item
        assert item in buffer
    
    raise NotImplementedError(f"Test that the ReplayBuffer works even with spaces that are a bit more complex.")