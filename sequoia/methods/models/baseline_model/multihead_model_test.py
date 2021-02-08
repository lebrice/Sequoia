"""Tests for the class-incremental version of the Model class.
"""
# from sequoia.conftest import config
from typing import Dict, List, Tuple, Type
from gym import spaces
import pytest
import torch
from sequoia.common.config import Config
from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import TaskSet
from sequoia.settings import ClassIncrementalSetting
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from sequoia.utils import take
from sequoia.common import Loss
from .multihead_model import MultiHeadModel, OutputHead


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
    def __init__(self, *args, Actions: Type, task_id: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.Actions = Actions
        self.name = f"task_{task_id}"

    def forward(self, observations, representations) -> Tensor:  # type: ignore
        x: Tensor = observations.x
        h_x = representations
        # actions = torch.stack([h_i.mean() * self.task_id for h_i in h_z])
        # actions = torch.stack([x_i.mean() * self.task_id for x_i in x])
        actions = [x_i.mean() * self.task_id  for x_i in x]
        actions = torch.stack(actions)
        return self.Actions(actions)
    
    def get_loss(self, forward_pass, actions, rewards):
        return Loss(self.name, 0.)
    
# def mock_output_task(self: MultiHeadModel, x: Tensor, h_x: Tensor) -> Tensor:
#     return self.output_head(x)

# def mock_encoder(self: MultiHeadModel, x: Tensor) -> Tensor:
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
    model = MultiHeadModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(batch_size=30, multihead=True),
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
        input_space=spaces.Box(0, 1, [model.hidden_size]),
        Actions=setting.Actions,
        action_space=spaces.Discrete(2),
        task_id=None,
    )
    for i in range(5):
        model.output_heads[str(i)] = MockOutputHead(
            input_space=spaces.Box(0, 1, [model.hidden_size]),
            Actions=setting.Actions,
            action_space=spaces.Discrete(2),
            task_id=i,
        )
    
    xs, ys, ts = map(torch.cat, zip(*mixed_samples.values()))
    
    xs = xs[indices]
    ys = ys[indices]
    ts = ts[indices].int()
    
    obs = setting.Observations(x=xs, task_labels=ts)
    with torch.no_grad():
        forward_pass = model(obs)
        y_preds = forward_pass["y_pred"]

    assert y_preds.shape == ts.shape
    assert torch.all(y_preds == ts * xs.view([xs.shape[0], -1]).mean(1))
    
    # Test that the output head predictions make sense:
    # print(ts)
    # for x, y_pred, task_id in zip(xs, y_preds, ts):
    #     assert y_pred.tolist() == (x.mean() * task_id).tolist()
        # assert y_pred.tolist() == (x.mean() * task_id).tolist() 
    
    # assert False, y_preds[0]
    
    # assert False, {i: [vi.shape for vi in v] for i, v in mixed_samples.items()}


def test_applied_to_multitask_rl():
    """ TODO: on_task_switch is called on the new observation, but we need to produce a
    loss for the output head that we were just using!
    """
    import gym
    from sequoia.common.gym_wrappers import MultiTaskEnvironment
    from gym.wrappers import TimeLimit
    env = gym.make("CartPole-v0")
    env = TimeLimit(env, max_episode_steps=10)
    env = MultiTaskEnvironment(
        env,
        task_schedule={
            0: {"length": 0.1},
            5: {"length": 0.2},
            200: {"length": 0.3},
            300: {"length": 0.4},
            400: {"length": 0.5},
        },
        add_task_id_to_obs=True,
        new_random_task_on_reset=True,
    )
    
    # Episodes only last 10 steps. Tasks don't have anything to do with the task
    # schedule.
    obs = env.reset()
    start_task_label = obs[1]
    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs[1] == start_task_label
        if i == 9:
            assert done
        else:
            assert not done
