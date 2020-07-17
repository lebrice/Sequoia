"""Utility functions used to manipulate generators.
"""
from .environment_test import DummyEnvironment
from .env_utils import ZipEnvironments


def test_zip_active_environments():
    env = ZipEnvironments(DummyEnvironment(0), DummyEnvironment(1))
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
