from functools import partial
from typing import Optional

import gym
import numpy as np
import pytest
import torch

from common.loss import Loss
from common.gym_wrappers.batch_env import BatchedVectorEnv
from conftest import DummyEnvironment
from gym import spaces
from torch import Tensor
from methods.models.forward_pass import ForwardPass
from settings.active.rl.continual_rl_setting import ContinualRLSetting

from .policy_head import PolicyHead

from common.gym_wrappers.batch_env.worker import FINAL_STATE_KEY
from common.gym_wrappers import AddDoneToObservation, ConvertToFromTensors, EnvDataset


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_loss_is_nonzero_at_episode_end(batch_size: int):
    """ Test that when stepping through the env, when the episode ends, a
    non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(temp_env, "reward_space",
                               spaces.Box(*temp_env.reward_range, shape=())) 

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = PolicyHead(
        observation_space=obs_space,
        representation_space=obs_space[0],
        action_space=action_space,
        reward_space=reward_space,
    )

    env.seed(123)
    obs = env.reset()

    # obs = torch.as_tensor(obs, dtype=torch.float32)

    done = torch.zeros(batch_size, dtype=bool)
    info = np.array([{} for _ in range(batch_size)])
    loss = None
    
    non_zero_losses = 0
    
    for i in range(100):
        representations = obs[0]
        observations = ContinualRLSetting.Observations(
            x=obs[0],
            done=done,
            # info=info,
        )
        head_output = head.forward(observations, representations=representations)
        actions = head_output.actions.numpy().tolist()
        # actions = np.zeros(batch_size, dtype=int).tolist()

        obs, rewards, done, info = env.step(actions)
        done = torch.as_tensor(done, dtype=bool)
        rewards = ContinualRLSetting.Rewards(rewards)
        assert len(info) == batch_size

        print(f"Step {i}, obs: {obs}, done: {done}, info: {info}")

        forward_pass = ForwardPass(         
            observations=observations,
            representations=representations,
            actions=head_output,
        )
        loss = head.get_loss(forward_pass, actions=head_output, rewards=rewards)
        print("loss:", loss)

        for env_index, env_is_done in enumerate(observations.done):
            if env_is_done:
                print(f"Episode ended for env {env_index} at step {i}")
                assert loss.total_loss != 0.
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss.total_loss == 0.

    assert non_zero_losses > 0


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_done_is_sometimes_True_when_iterating_through_env(batch_size: int):
    """ Test that when *iterating* through the env, done is sometimes 'True'.
    """
    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=True)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)
    for i, obs in zip(range(100), env):
        print(i, obs[1])
        reward = env.send(env.action_space.sample())
        if any(obs[1]):
            break
    else:
        assert False, "Never encountered done=True!"



@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_loss_is_nonzero_at_episode_end_iterate(batch_size: int):
    """ Test that when *iterating* through the env (active-dataloader style),
    when the episode ends, a non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)
        
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(temp_env, "reward_space",
                               spaces.Box(*temp_env.reward_range, shape=())) 

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = PolicyHead(
        observation_space=obs_space,
        representation_space=obs_space[0],
        action_space=action_space,
        reward_space=reward_space,
    )

    env.seed(123)
    non_zero_losses = 0
    
    for i, obs in zip(range(100), env):
        
        done = obs[1]
        x = obs[0]
        representations = x
        assert isinstance(x, Tensor)
        assert isinstance(done, Tensor)
        observations = ContinualRLSetting.Observations(
            x=x,
            done=done,
            # info=info,
        )
        head_output = head.forward(observations, representations=representations)
        
        actions = head_output.actions.numpy().tolist()
        # actions = np.zeros(batch_size, dtype=int).tolist()

        rewards = env.send(actions) 

        print(f"Step {i}, obs: {obs}, done: {done}")
        assert isinstance(representations, Tensor)
        forward_pass = ForwardPass(         
            observations=observations,
            representations=representations,
            actions=head_output,
        )
        rewards = ContinualRLSetting.Rewards(rewards)
        loss = head.get_loss(forward_pass, actions=head_output, rewards=rewards)
        print("loss:", loss)

        for env_index, env_is_done in enumerate(observations.done):
            if env_is_done:
                print(f"Episode ended for env {env_index} at step {i}")
                assert loss.total_loss != 0.
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss.total_loss == 0.

    assert non_zero_losses > 0

def test_buffers_are_stacked_correctly(monkeypatch):
    """TODO: Test that when "de-synced" episodes, when fed to the output head,
    get passed, re-stacked correctly, to the get_episode_loss function.
    """
    batch_size = 5
    
    starting_values = [i for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]
    
    env = BatchedVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
        for start, target in zip(starting_values, targets)
    ])
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))
    
    reward_space = spaces.Box(*env.reward_range, shape=())
    
    output_head = PolicyHead(1, action_space=env.single_action_space, reward_space=reward_space)
    # Set the max window length, for testing.
    output_head.hparams.max_episode_window_length = 100
    
    obs = initial_obs = env.reset()
    done = np.zeros(batch_size, dtype=bool)

    obs = torch.from_numpy(obs)
    done = torch.from_numpy(done)
    
    def mock_get_episode_loss(self: PolicyHead,
                              env_index: int,
                              observations: ContinualRLSetting.Observations,
                              actions: ContinualRLSetting.Observations,
                              rewards: ContinualRLSetting.Rewards,
                              episode_ended: bool) -> Optional[Loss]:
        print(f"Environment at index {env_index}, episode ended: {episode_ended}")
        if episode_ended:
            print(f"Full episode: {observations}")
        else:
            print(f"Episode so far: {observations}")
        
        n_observations = len(observations.x)
        # assert len(actions.y_pred) == n_observations if not episode_ended else n_observations - 1, (episode_ended, observations, actions, rewards)
        # assert len(rewards.y) == n_observations if not episode_ended else n_observations - 1, (episode_ended, observations, actions, rewards)

        assert observations.x.tolist() == (env_index + np.arange(n_observations)).tolist()
        if episode_ended:
            # Unfortunately, we don't get the final state, because of how
            # VectorEnv works atm.
            assert observations.x[-1] == targets[env_index] - 1
        
    monkeypatch.setattr(PolicyHead, "get_episode_loss", mock_get_episode_loss)

    # perform 10 iterations, incrementing each DummyEnvironment's counter at
    # each step (action of 1).
    # Therefore, at first, the counters should be [0, 1, 2, ... batch-size-1].
    info = [{} for _ in range(batch_size)]
    
    for step in range(10):
        print(f"Step {step}.")
        # Wrap up the obs to pretend that this is the data coming from a
        # ContinualRLSetting.
        observations = ContinualRLSetting.Observations(x=obs, done=done)#, info=info)
        # We don't use an encoder for testing, so the representations is just x.
        representations = obs.reshape([batch_size, 1])
        assert observations.task_labels is None

        # Instead of actually getting the action from the output head, we just
        # set the action, so all that we're testing is the loss part.

        # actions = output_head(observations, representations)
        actions = ContinualRLSetting.Actions(y_pred=torch.ones(batch_size, dtype=int))

        # Wrap things up to pretend like the output head is being used in the
        # BaselineModel:
        forward_pass = ForwardPass(
            observations = observations,
            representations = representations,
            actions = actions,
        )

        action_np = actions.actions_np
        
        obs, rewards, done, info = env.step(action_np)
        
        obs = torch.from_numpy(obs)
        rewards = torch.from_numpy(rewards)
        done = torch.from_numpy(done)
        
        rewards = ContinualRLSetting.Rewards(y=rewards)
        loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)
        
        # Check the contents of the episode buffers.

        assert len(output_head.episode_buffers) == batch_size
        for env_index, env_episode_buffer in enumerate(output_head.episode_buffers):
            if step >= batch_size:
                if step + env_index == targets[env_index]:
                    assert not env_episode_buffer
                # if env_index == step - batch_size:
                continue
            assert len(env_episode_buffer) == step + 1
            # Check to see that the last entry in the episode buffer for this
            # environment corresponds to the slice of the most recent
            # observations/actions/rewards at the index corresponding to this
            # environment.
            observation_tuple, action_tuple, reward_tuple = env_episode_buffer[-1]

            assert observation_tuple.x == observations.x[env_index]
            assert observation_tuple.task_labels is None
            assert observation_tuple.done == observations.done[env_index]

            assert action_tuple.y_pred == actions.y_pred[env_index]

            assert reward_tuple.y == rewards.y[env_index]

        if step < batch_size:
            assert obs.tolist() == (np.arange(batch_size) + step + 1).tolist()
        # if step >= batch_size:
        #     if step + env_index == targets[env_index]:
        #         assert done
                
    # assert False, (obs, rewards, done, info)
    # loss: Loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)



def test_sanity_check_cartpole_done_vector(monkeypatch):
    """TODO: Sanity check, make sure that cartpole has done=True at some point
    when using a BatchedEnv.
    """
    batch_size = 5
    
    starting_values = [i for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]
    from settings.active.rl.make_env import make_batched_env
    from common.gym_wrappers import PixelObservationWrapper
    from settings.active.rl.wrappers import AddDoneToObservation, AddInfoToObservation
    env = make_batched_env("CartPole-v0", batch_size=5, wrappers=[PixelObservationWrapper])
    env = AddDoneToObservation(env)
    env = AddInfoToObservation(env)
    
    # env = BatchedVectorEnv([
    #     partial(gym.make, "CartPole-v0") for i in range(batch_size)
    # ])
    obs = env.reset()
    
    for i in range(100):
        obs, rewards, done, info = env.step(env.action_space.sample())
        assert all(obs[1] == done), i
        if any(done):
            
            break
    else:
        assert False, "Should have had at least one done=True, over the 100 steps!"