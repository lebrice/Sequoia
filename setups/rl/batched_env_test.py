"""Utility functions used to manipulate generators.
"""
from ..environment_test import DummyEnvironment
from .batched_env import BatchEnvironments


def test_zip_active_environments():
    env = BatchEnvironments(DummyEnvironment(0), DummyEnvironment(1))
    x = next(env)
    assert x == [0, 1]
    for i, x in enumerate(env):
        assert x == [i+1, i+2]
        if i == 3:
            break
    assert x == [4, 5]
    
    env.send([0, 2])
    x = next(env)
    assert x == [5, 8] 



def test_zip_method_call():
    env = BatchEnvironments(DummyEnvironment(0), DummyEnvironment(1))
    print(env.peek())
    assert env.peek() == [0, 1]
    assert env.add([2, 1]) == [2, 2]
    env.reset()
    assert env.peek() == [0, 0]


from setups.environment_test import DummyEnvironment


def test_zip_method_call():
    env = BatchEnvironments(DummyEnvironment(0), DummyEnvironment(1))
    assert env.peek() == [0, 1]
    assert env.add([2, 1]) == [2, 2]
    env.reset()
    assert env.peek() == [0, 0]
