import pytest
import torch
from torch.utils.data import DataLoader

from settings.active.active_dataloader_test import DummyEnvironment

from .zip_dataset import ZipDataset


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

def collate_fn(tensors):
    print(f"inside collate_fn: {tensors}")
    return torch.stack([torch.as_tensor(v) for v in tensors])

from ..active_dataloader import ActiveDataLoader

@pytest.mark.xfail(reason="Don't yet have multi-worker active dataloader working.")
@pytest.mark.parametrize("n_workers", [0, 1, 2, 4, 8, 24])
def test_zip_dataset_multiple_workers(n_workers):
    # TODO: Test that the ZipDataset actually works with multiple workers.
    # Not sure how to test this properly..
    n_environments: int = max(n_workers, 1)
    dataset = ZipDataset([DummyEnvironment(i * 10) for i in range(n_environments)])
    dataloader = ActiveDataLoader(
        dataset,
        num_workers=n_workers,
        batch_size=None,
        collate_fn=collate_fn,
    )
    batch_size = n_environments

    for i, batch in enumerate(dataloader):
        batch = torch.as_tensor(batch)
        assert batch.tolist() == list(range(i, n_environments * 10 + i, 10)), i
        assert batch.shape == (batch_size,)
        
        reward = dataloader.send([0 for _ in range(n_environments)])
        
        if i >= 1:
            break