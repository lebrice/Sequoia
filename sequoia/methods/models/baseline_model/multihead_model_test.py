"""Tests for the class-incremental version of the Model class.
"""
# from sequoia.conftest import config
from typing import Dict, List, Tuple, Type

import pytest
import torch
from collections import defaultdict
from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import TaskSet
from gym import spaces
from gym.spaces.utils import flatdim
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from sequoia.common import Loss
from sequoia.common.config import Config
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.base import Environment
from sequoia.utils import take
from sequoia.methods.models.forward_pass import ForwardPass

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

    # model.output_head = MockOutputHead(
    #     input_space=spaces.Box(0, 1, [model.hidden_size]),
    #     Actions=setting.Actions,
    #     action_space=spaces.Discrete(2),
    #     task_id=None,
    # )
    for i in range(5):
        model.output_heads[str(i)] = MockOutputHead(
            input_space=spaces.Box(0, 1, [model.hidden_size]),
            Actions=setting.Actions,
            action_space=spaces.Discrete(2),
            task_id=i,
        )
    model.output_head = model.output_heads["0"]
    
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

import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import TimeLimit
from sequoia.common.gym_wrappers import MultiTaskEnvironment
from sequoia.settings import RLSetting


def get_multi_task_env(batch_size: int = 1) -> Environment[RLSetting.Observations, RLSetting.Actions, RLSetting.Rewards]:
    def single_env_fn() -> gym.Env:
        env = gym.make("CartPole-v0")
        env = TimeLimit(env, max_episode_steps=10)
        env = MultiTaskEnvironment(
            env,
            task_schedule={
                0: {"length": 0.1},
                100: {"length": 0.2},
                200: {"length": 0.3},
                300: {"length": 0.4},
                400: {"length": 0.5},
            },
            add_task_id_to_obs=True,
            new_random_task_on_reset=True,
        )
        return env

    batch_size = 1
    env = SyncVectorEnv([single_env_fn for _ in range(batch_size)])
    from sequoia.common.gym_wrappers import AddDoneToObservation
    from sequoia.settings.active import TypedObjectsWrapper
    env = AddDoneToObservation(env)
    # Wrap the observations so they appear as though they are from the given setting.
    env = TypedObjectsWrapper(
        env,
        observations_type=RLSetting.Observations,
        actions_type=RLSetting.Actions,
        rewards_type=RLSetting.Rewards,
    )
    env.seed(123)
    return env

from .baseline_model import BaselineModel
from sequoia.methods.models.output_heads.rl.episodic_a2c import EpisodicA2C


def test_multitask_rl_bug_without_PL(monkeypatch):
    """ TODO: on_task_switch is called on the new observation, but we need to produce a
    loss for the output head that we were just using!
    """
    # NOTE: Tasks don't have anything to do with the task schedule. They are sampled at
    # each episode.
    max_episode_steps = 5
    setting = RLSetting(
        dataset="cartpole",
        batch_size=1,
        nb_tasks=2,
        max_episode_steps=max_episode_steps,
        add_done_to_observations=True,
        observe_state_directly=True,
    )
    assert setting._new_random_task_on_reset
    
    # setting = RLSetting.load_benchmark("monsterkong")
    config = Config(debug=True, verbose=True, seed=123)
    config.seed_everything()
    model = BaselineModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(multihead=True, output_head=EpisodicA2C.HParams(accumulate_losses_before_backward=True)),
        config=config,
    )
    # TODO: Maybe add some kind of "hook" to check which losses get returned when?
    model.train()
    # from pytorch_lightning import Trainer
    # trainer = Trainer(fast_dev_run=True)
    # trainer.fit(model, train_dataloader=setting.train_dataloader())
    # trainer.setup(model, stage="fit")
    
    # from pytorch_lightning import Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    episodes = 0
    max_episodes = 5
    
    # Dict mapping from step to loss at that step.
    losses: Dict[int, Loss] = {} 
    
    with setting.train_dataloader() as env:
        env.seed(123)
        # env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Iterate over the environment, which yields one observation at a time:
        for step, obs in enumerate(env):
            assert isinstance(obs, RLSetting.Observations)

            if step == 0:
                assert not any(obs.done)
            start_task_label = obs[1][0]
            
            stored_steps_in_each_head_before = {
                task_key: output_head.num_stored_steps(0)
                for task_key, output_head in model.output_heads.items()
            }
            forward_pass: ForwardPass = model.forward(observations=obs)
            rewards = env.send(forward_pass.actions)

            loss: Loss = model.get_loss(forward_pass=forward_pass, rewards=rewards, loss_name="debug")
            stored_steps_in_each_head_after = {
                task_key: output_head.num_stored_steps(0)
                for task_key, output_head in model.output_heads.items()
            }
            # if step == 5:
            #     assert False, (loss, stored_steps_in_each_head_before, stored_steps_in_each_head_after)

            if any(obs.done):
                assert loss.loss != 0., step
                assert loss.loss.requires_grad
                
                # Backpropagate the loss, update the models, etc etc.
                loss.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                episodes += sum(obs.done)
                losses[step] = loss
            else:
                assert loss.loss == 0.
            # TODO: 
            print(f"Step {step}, episode {episodes}: x={obs[0]}, done={obs.done}, reward={rewards} task labels: {obs.task_labels}, loss: {loss.losses.keys()}: {loss.loss}")

            if episodes > max_episodes:
                break
    # assert False, losses
    


def test_multitask_rl_bug_with_PL(monkeypatch):
    """ TODO: on_task_switch is called on the new observation, but we need to produce a
    loss for the output head that we were just using!
    """
    # NOTE: Tasks don't have anything to do with the task schedule. They are sampled at
    # each episode.
    max_episode_steps = 5
    setting = RLSetting(
        dataset="cartpole",
        batch_size=1,
        nb_tasks=2,
        max_episode_steps=max_episode_steps,
        add_done_to_observations=True,
        observe_state_directly=True,
    )
    assert setting._new_random_task_on_reset
    
    # setting = RLSetting.load_benchmark("monsterkong")
    config = Config(debug=True, verbose=True, seed=123)
    config.seed_everything()
    model = BaselineModel(
        setting=setting,
        hparams=MultiHeadModel.HParams(multihead=True, output_head=EpisodicA2C.HParams(accumulate_losses_before_backward=True)),
        config=config,
    )

    # TODO: Maybe add some kind of "hook" to check which losses get returned when?
    model.train()
    assert not model.automatic_optimization
    
    from pytorch_lightning import Trainer
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloader=setting.train_dataloader())
    
    # from pytorch_lightning import Trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    episodes = 0
    max_episodes = 5
    
    # Dict mapping from step to loss at that step.
    losses: Dict[int, List[Loss]] = defaultdict(list)
    
    with setting.train_dataloader() as env:
        env.seed(123)
        
        
        # env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Iterate over the environment, which yields one observation at a time:
        for step, obs in enumerate(env):
            assert isinstance(obs, RLSetting.Observations)
            
            step_results = model.training_step(
                    batch=obs,
                    batch_idx=step
            )
            loss_tensor: Optional[Tensor] = None
            
            if step > 0 and step % 5 == 0:
                assert all(obs.done), step # Since batch_size == 1 for now.
                assert step_results is not None, (step, obs.task_labels)
                loss_tensor = step_results["loss"]
                loss: Loss = step_results["loss_object"]
                print(f"Loss at step {step}: {loss}")
                losses[step].append(loss)
                
                # # Manually perform the optimization step.
                # output_head_loss = loss.losses.get(model.output_head.name)            
                # update_model = output_head_loss is not None and output_head_loss.requires_grad

                # assert update_model
                # model.manual_backward(loss_tensor, optimizer, retain_graph=not update_model)
                # model.optimizer_step()
                # if update_model:
                #     optimizer.step()
                #     optimizer.zero_grad()
                # else:
                #     assert False, (loss, output_head_loss, model.output_head.name)

            else:
                assert step_results is None

            print(f"Step {step}, episode {episodes}: x={obs[0]}, done={obs.done}, task labels: {obs.task_labels}, loss_tensor: {loss_tensor}")    
            
            if step > 100:
                break

    for step, step_losses in losses.items():
        print(f"Losses at step {step}:")
        for loss in step_losses:
            print(f"\t{loss}")
    # assert False, losses