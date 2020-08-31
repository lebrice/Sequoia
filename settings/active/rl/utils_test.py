from .utils import ZipDataset
from settings.active.active_dataloader_test import DummyEnvironment


def test_zip_passive_datasets():
    env = ZipDataset([DummyEnvironment(0), DummyEnvironment(1)])
    x = next(env)
    assert x == [0, 1]
    for i, x in enumerate(env):
        assert x == [i+1, i+2]
        if i == 3:
            break
    assert x == [4, 5]


def test_zip_active_datasets():
    env = ZipDataset([DummyEnvironment(0), DummyEnvironment(1)])
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
