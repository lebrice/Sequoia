"""Tests for the class-incremental version of the Model class.
"""
# from conftest import config
from typing import Dict, List, Tuple, Type
from gym import spaces
import pytest
import torch
from common.config import Config
from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import TaskSet
from settings import ClassIncrementalSetting
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from utils import take

from .class_incremental_model import ClassIncrementalModel, OutputHead


@pytest.fixture()
def mixed_samples(config: Config):
    """ Fixture that produces some samples from each task. """
    dataset = MNIST(config.data_dir, download=True, train=True)
    datasets: List[TaskSet] = ClassIncremental(dataset, nb_tasks=5)
    n_samples_per_task = 10
    indices = list(range(10))
    samples_per_task: Dict[int, Tensor] = {
        i: tuple(map(torch.as_tensor, taskset.get_samples(indices)))
        for i, taskset in enumerate(datasets)
    }
    yield samples_per_task


class MockOutputHead(OutputHead):
    def __init__(self, input_size: int, Actions: Type, task_id: int = -1, **kwargs):
        self.task_id = task_id
        self.Actions = Actions
        super().__init__(input_size=input_size, **kwargs)
        

    def forward(self, observations, representations) -> Tensor:  # type: ignore
        x: Tensor = observations.x
        h_x = representations
        # TODO: We should maybe convert this to also return a dict instead
        # of a Tensor, just to be consistent with everything else. This could
        # also maybe help with having multiple different output heads, each
        # having a different name and giving back a dictionary of their own
        # forward pass tensors (if needed) and predictions?
        actions = torch.stack([x_i.mean() * self.task_id for x_i in x])
        return self.Actions(actions)
# def mock_output_task(self: ClassIncrementalModel, x: Tensor, h_x: Tensor) -> Tensor:
#     return self.output_head(x)

# def mock_encoder(self: ClassIncrementalModel, x: Tensor) -> Tensor:
#     return x.new_ones(self.hp.hidden_size)


@pytest.mark.parametrize("indices", [
    slice(0, 10), # all the same task (0)
    slice(0, 20), # 10 from task 0, 10 from task 1
    slice(0, 30), # 10 from task 0, 10 from task 1, 10 from task 2
    slice(0, 50), # 10 from each task.
])
def test_multiple_tasks_within_same_batch(mixed_samples: Dict[int, Tuple[Tensor, Tensor, Tensor]],
                                          indices: slice,
                                          monkeypatch, config: Config):
    """ TODO: Write out a test that checks that when given a batch with data
    from different tasks, and when the model is multiheaded, it will use the
    right output head for each image.
    """
    setting = ClassIncrementalSetting()
    model = ClassIncrementalModel(
        setting=setting,
        hparams=ClassIncrementalModel.HParams(batch_size=30, multihead=True),
        config=config,
    )
    
    class MockEncoder(nn.Module):
        def forward(self, x: Tensor):
            return x.new_ones([x.shape[0], model.hidden_size])

    mock_encoder = MockEncoder()
    # monkeypatch.setattr(model, "forward", mock_encoder_forward)
    model.encoder = mock_encoder
    # model.output_task = mock_output_task
    model.output_head = MockOutputHead(
        input_size=model.hidden_size, 
        Actions=setting.Actions,
        action_space=spaces.Discrete(2),
        task_id=None,
    )
    for i in range(5):
        model.output_heads[str(i)] = MockOutputHead(
            input_size=model.hidden_size,
            Actions=setting.Actions,
            action_space=spaces.Discrete(2),
            task_id=i,
        )
    
    xs, ys, ts = map(torch.cat, zip(*mixed_samples.values()))
    
    
    images = xs[indices]
    labels = ys[indices]
    task_ids = ts[indices].int()
    
    obs = setting.Observations(x=images, task_labels=task_ids)
    # assert False, obs
    with torch.no_grad():
        forward_pass = model(obs)
        y_preds = forward_pass["y_pred"]
    
    for x, y_pred, task_id in zip(xs, y_preds, task_ids):
        # print(y_pred)
        # print(x.mean() * task_id)
        assert y_pred == x.mean() * task_id 
    
    # assert False, y_preds[0]
    
    # assert False, {i: [vi.shape for vi in v] for i, v in mixed_samples.items()}
