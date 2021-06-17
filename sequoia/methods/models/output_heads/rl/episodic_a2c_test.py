from functools import partial
from typing import Callable, Optional, Sequence

import gym
import numpy as np
import pytest
import torch
from gym import spaces
from gym.spaces.utils import flatdim
from gym.vector import SyncVectorEnv
from gym.vector.utils import batch_space
from sequoia.common.gym_wrappers import (
    AddDoneToObservation,
    ConvertToFromTensors,
    EnvDataset,
)
from sequoia.common.gym_wrappers.batch_env import BatchedVectorEnv
from sequoia.common.loss import Loss
from sequoia.conftest import DummyEnvironment
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.settings.rl.continual import ContinualRLSetting
from torch import Tensor, nn

from .episodic_a2c import A2CHeadOutput, EpisodicA2C
from .policy_head import PolicyHead


class FakeEnvironment(SyncVectorEnv):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        batch_size: int,
        new_episode_length: Callable[[int], int],
        episode_lengths: Sequence[int] = None,
    ):
        super().__init__([env_fn for _ in range(batch_size)])
        self.new_episode_length = new_episode_length
        self.batch_size = batch_size
        self.episode_lengths = np.array(
            episode_lengths or [new_episode_length(i) for i in range(self.num_envs)]
        )
        self.steps_left_in_episode = self.episode_lengths.copy()

        reward_space = spaces.Box(*self.reward_range, shape=())
        self.single_reward_space = reward_space
        self.reward_space = batch_space(reward_space, batch_size)

    def step(self, actions):
        self.steps_left_in_episode[:] -= 1

        # obs, reward, done, info = super().step(actions)
        obs = self.observation_space.sample()
        reward = np.ones(self.batch_size)

        assert not any(self.steps_left_in_episode < 0)
        done = self.steps_left_in_episode == 0

        info = np.array([{} for _ in range(self.batch_size)])

        for env_index, env_done in enumerate(done):
            if env_done:
                next_episode_length = self.new_episode_length(env_index)
                self.episode_lengths[env_index] = next_episode_length
                self.steps_left_in_episode[env_index] = next_episode_length

        return obs, reward, done, info


@pytest.mark.xfail(
    reason="TODO: Adapt this test for EpisodicA2C (copied form policy_head_test.py)"
)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_with_controllable_episode_lengths(batch_size: int, monkeypatch):
    """ TODO: Test out the EpisodicA2C output head in a very controlled environment,
    where we know exactly the lengths of each episode.
    """
    env = FakeEnvironment(
        partial(gym.make, "CartPole-v0"),
        batch_size=batch_size,
        episode_lengths=[5, *(10 for _ in range(batch_size - 1))],
        new_episode_length=lambda env_index: 10,
    )
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    obs_space = env.single_observation_space
    x_dim = flatdim(obs_space["x"])
    # Create some dummy encoder.
    encoder = nn.Linear(x_dim, x_dim)
    representation_space = obs_space["x"]

    output_head = EpisodicA2C(
        input_space=representation_space,
        action_space=env.single_action_space,
        reward_space=env.single_reward_space,
        hparams=PolicyHead.HParams(
            max_episode_window_length=100,
            min_episodes_before_update=1,
            accumulate_losses_before_backward=False,
        ),
    )
    # TODO: Simplify the loss function somehow using monkeypatch so we know exactly what
    # the loss should be at each step.

    batch_size = env.batch_size

    obs = env.reset()
    step_done = np.zeros(batch_size, dtype=np.bool)

    for step in range(200):
        x, obs_done = obs

        # The done from the obs should always be the same as the 'done' from the 'step' function.
        assert np.array_equal(obs_done, step_done)

        representations = encoder(x)
        observations = ContinualRLSetting.Observations(x=x, done=obs_done,)

        actions_obj = output_head(observations, representations)
        actions = actions_obj.y_pred

        # TODO: kinda useless to wrap a single tensor in an object..
        forward_pass = ForwardPass(
            observations=observations, representations=representations, actions=actions,
        )
        obs, rewards, step_done, info = env.step(actions)

        rewards_obj = ContinualRLSetting.Rewards(y=rewards)
        loss = output_head.get_loss(
            forward_pass=forward_pass, actions=actions_obj, rewards=rewards_obj,
        )
        print(f"Step {step}")
        print(f"num episodes since update: {output_head.num_episodes_since_update}")
        print(f"steps left in episode: {env.steps_left_in_episode}")
        print(f"Loss for that step: {loss}")

        if any(obs_done):
            assert loss != 0.0

        if step == 5.0:
            # Env 0 first episode from steps 0 -> 5
            assert loss.loss == 5.0
            assert loss.metrics["gradient_usage"].used_gradients == 5.0
            assert loss.metrics["gradient_usage"].wasted_gradients == 0.0
        elif step == 10:
            # Envs[1:batch_size], first episode, from steps 0 -> 10
            # NOTE: At this point, both envs have reached the required number of episodes.
            # This means that the gradient usage on the next time any env reaches
            # an end-of-episode will be one less than the total number of items.
            assert loss.loss == 10.0 * (batch_size - 1)
            assert loss.metrics["gradient_usage"].used_gradients == 10.0 * (
                batch_size - 1
            )
            assert loss.metrics["gradient_usage"].wasted_gradients == 0.0
        elif step == 15:
            # Env 0 second episode from steps 5 -> 15
            assert loss.loss == 10.0
            assert loss.metrics["gradient_usage"].used_gradients == 4
            assert loss.metrics["gradient_usage"].wasted_gradients == 6

        elif step == 20:
            # Envs[1:batch_size]: second episode, from steps 0 -> 10
            # NOTE: At this point, both envs have reached the required number of episodes.
            # This means that the gradient usage on the next time any env reaches
            # an end-of-episode will be one less than the total number of items.
            assert loss.loss == 10.0 * (batch_size - 1)
            assert loss.metrics["gradient_usage"].used_gradients == 9 * (batch_size - 1)
            assert loss.metrics["gradient_usage"].wasted_gradients == 1 * (
                batch_size - 1
            )

        elif step == 25:
            # Env 0 third episode from steps 5 -> 15
            assert loss.loss == 10.0
            assert loss.metrics["gradient_usage"].used_gradients == 4
            assert loss.metrics["gradient_usage"].wasted_gradients == 6

        elif step > 0 and step % 10 == 0:
            # Same pattern as step 20 above
            assert loss.loss == 10.0 * (batch_size - 1), step
            assert loss.metrics["gradient_usage"].used_gradients == 9 * (batch_size - 1)
            assert loss.metrics["gradient_usage"].wasted_gradients == 1 * (
                batch_size - 1
            )

        elif step > 0 and step % 5 == 0:
            # Same pattern as step 25 above
            assert loss.loss == 10.0
            assert loss.metrics["gradient_usage"].used_gradients == 4
            assert loss.metrics["gradient_usage"].wasted_gradients == 6

        else:
            assert loss.loss == 0.0, step


@pytest.mark.parametrize("batch_size", [1, 2, 5,])
def test_loss_is_nonzero_at_episode_end(batch_size: int):
    """ Test that when stepping through the env, when the episode ends, a
    non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(
            temp_env, "reward_space", spaces.Box(*temp_env.reward_range, shape=())
        )

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = EpisodicA2C(
        input_space=obs_space["x"],
        action_space=action_space,
        reward_space=reward_space,
        hparams=EpisodicA2C.HParams(accumulate_losses_before_backward=False),
    )
    head.train()

    env.seed(123)
    obs = env.reset()

    # obs = torch.as_tensor(obs, dtype=torch.float32)

    done = torch.zeros(batch_size, dtype=bool)
    info = np.array([{} for _ in range(batch_size)])
    loss = None

    non_zero_losses = 0

    encoder = nn.Linear(4, 4)
    encoder.train()

    for i in range(100):
        representations = encoder(obs["x"])

        observations = ContinualRLSetting.Observations(
            x=obs["x"],
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

        assert observations.done is not None
        for env_index, env_is_done in enumerate(observations.done):
            if env_is_done:
                print(f"Episode ended for env {env_index} at step {i}")
                assert loss.loss != 0.0
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss is None or loss.loss == 0.0

    assert non_zero_losses > 0



@pytest.mark.xfail(
    reason="TODO: Adapt this test for EpisodicA2C (copied form policy_head_test.py)"
)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_loss_is_nonzero_at_episode_end_iterate(batch_size: int):
    """ Test that when *iterating* through the env (active-dataloader style),
    when the episode ends, a non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)

        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(
            temp_env, "reward_space", spaces.Box(*temp_env.reward_range, shape=())
        )

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = EpisodicA2C(
        # observation_space=obs_space,
        input_space=obs_space["x"],
        action_space=action_space,
        reward_space=reward_space,
        hparams=EpisodicA2C.HParams(accumulate_losses_before_backward=False),
    )

    env.seed(123)
    non_zero_losses = 0

    for i, obs in zip(range(100), env):
        print(i, obs)
        x = obs["x"]
        done = obs[1]
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

        # print(f"Step {i}, obs: {obs}, done: {done}")
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
                assert loss.total_loss != 0.0
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss.total_loss == 0.0

    assert non_zero_losses > 0


@pytest.mark.xfail(
    reason="TODO: Adapt this test for EpisodicA2C (copied form policy_head_test.py)"
)
@pytest.mark.xfail(reason="TODO: Fix this test")
def test_buffers_are_stacked_correctly(monkeypatch):
    """TODO: Test that when "de-synced" episodes, when fed to the output head,
    get passed, re-stacked correctly, to the get_episode_loss function.
    """
    batch_size = 5

    starting_values = [i for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]

    env = BatchedVectorEnv(
        [
            partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
            for start, target in zip(starting_values, targets)
        ]
    )
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))

    reward_space = spaces.Box(*env.reward_range, shape=())
    output_head = PolicyHead(  # observation_space=spaces.Tuple([env.observation_space,
        #              spaces.Box(False, True, [batch_size], np.bool)]),
        input_space=spaces.Box(0, 1, (1,)),
        action_space=env.single_action_space,
        reward_space=reward_space,
    )
    # Set the max window length, for testing.
    output_head.hparams.max_episode_window_length = 100

    obs = initial_obs = env.reset()
    done = np.zeros(batch_size, dtype=bool)

    obs = torch.from_numpy(obs)
    done = torch.from_numpy(done)

    def mock_get_episode_loss(
        self: PolicyHead,
        env_index: int,
        inputs: Tensor,
        actions: ContinualRLSetting.Observations,
        rewards: ContinualRLSetting.Rewards,
        done: bool,
    ) -> Optional[Loss]:
        print(f"Environment at index {env_index}, episode ended: {done}")
        if done:
            print(f"Full episode: {inputs}")
        else:
            print(f"Episode so far: {inputs}")

        n_observations = len(inputs)

        assert (
            inputs.flatten().tolist()
            == (env_index + np.arange(n_observations)).tolist()
        )
        if done:
            # Unfortunately, we don't get the final state, because of how
            # VectorEnv works atm.
            assert inputs[-1] == targets[env_index] - 1

    monkeypatch.setattr(PolicyHead, "get_episode_loss", mock_get_episode_loss)

    # perform 10 iterations, incrementing each DummyEnvironment's counter at
    # each step (action of 1).
    # Therefore, at first, the counters should be [0, 1, 2, ... batch-size-1].
    info = [{} for _ in range(batch_size)]

    for step in range(10):
        print(f"Step {step}.")
        # Wrap up the obs to pretend that this is the data coming from a
        # ContinualRLSetting.
        observations = ContinualRLSetting.Observations(x=obs, done=done)  # , info=info)
        # We don't use an encoder for testing, so the representations is just x.
        representations = obs.reshape([batch_size, 1])
        assert observations.task_labels is None

        actions = output_head(observations.float(), representations.float())

        # Wrap things up to pretend like the output head is being used in the
        # BaselineModel:

        forward_pass = ForwardPass(
            observations=observations, representations=representations, actions=actions,
        )

        action_np = actions.actions_np

        obs, rewards, done, info = env.step(action_np)

        obs = torch.from_numpy(obs)
        rewards = torch.from_numpy(rewards)
        done = torch.from_numpy(done)

        rewards = ContinualRLSetting.Rewards(y=rewards)
        loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)

        # Check the contents of the episode buffers.

        assert len(output_head.representations) == batch_size
        for env_index in range(batch_size):

            # obs_buffer = output_head.observations[env_index]
            representations_buffer = output_head.representations[env_index]
            action_buffer = output_head.actions[env_index]
            reward_buffer = output_head.rewards[env_index]

            if step >= batch_size:
                if step + env_index == targets[env_index]:
                    assert (
                        len(representations_buffer) == 1
                        and output_head.done[env_index] == False
                    )
                # if env_index == step - batch_size:
                continue
            assert len(representations_buffer) == step + 1
            # Check to see that the last entry in the episode buffer for this
            # environment corresponds to the slice of the most recent
            # observations/actions/rewards at the index corresponding to this
            # environment.

            # observation_tuple = input_buffer[-1]
            step_action = action_buffer[-1]
            step_reward = reward_buffer[-1]
            # assert observation_tuple.x == observations.x[env_index]
            # assert observation_tuple.task_labels is None
            # assert observation_tuple.done == observations.done[env_index]

            # The last element in the buffer should be the slice in the batch
            # for that environment.
            assert step_action.y_pred == actions.y_pred[env_index]
            assert step_reward.y == rewards.y[env_index]

        if step < batch_size:
            assert obs.tolist() == (np.arange(batch_size) + step + 1).tolist()
        # if step >= batch_size:
        #     if step + env_index == targets[env_index]:
        #         assert done

    # assert False, (obs, rewards, done, info)
    # loss: Loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)
