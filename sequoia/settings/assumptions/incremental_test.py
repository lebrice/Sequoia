from typing import List, Optional

import gym
import numpy as np
from gym import Space
from gym.vector.utils.spaces import batch_space
from sequoia.methods import Method
from sequoia.settings import Actions, Environment, Observations, Setting

from .incremental import IncrementalSetting, TestEnvironment


class DummyMethod(Method, target_setting=IncrementalSetting):
    """ Dummy method used to check that the Setting calls `on_task_switch` with the
    right arguments.
    """

    def __init__(self):
        self.n_task_switches = 0
        self.n_fit_calls = 0
        self.received_task_ids: List[Optional[int]] = []
        self.received_while_training: List[bool] = []
        self.train_steps_per_task: List[int] = []
        self.train_episodes_per_task: List[int] = []

    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        self.n_fit_calls += 1
        self.train_steps_per_task.append(0)
        self.train_episodes_per_task.append(0)
        obs = train_env.reset()
        for i in range(100):
            obs, reward, done, info = train_env.step(train_env.action_space.sample())
            self.train_steps_per_task[-1] += 1
            if done:
                self.train_episodes_per_task[-1] += 1
                break

    def test(self, test_env: TestEnvironment):
        while not test_env.is_closed():
            done = False
            obs = test_env.reset()
            while not done:
                actions = test_env.action_space.sample()
                obs, _, done, info = test_env.step(actions)

    def get_actions(
        self, observations: IncrementalSetting.Observations, action_space: gym.Space
    ):
        return np.ones(action_space.shape)

    def on_task_switch(self, task_id: int = None):
        self.n_task_switches += 1
        self.received_task_ids.append(task_id)
        self.received_while_training.append(self.training)


class OtherDummyMethod(Method, target_setting=IncrementalSetting):
    def __init__(self):
        self.batch_sizes: List[int] = []

    def fit(self, train_env: Environment, valid_env: Environment):
        for i, batch in enumerate(train_env):
            if isinstance(batch, Observations):
                observations, rewards = batch, None
            else:
                assert isinstance(batch, tuple) and len(batch) == 2
                observations, rewards = batch

            y_preds = train_env.action_space.sample()
            if rewards is None:
                action_space = train_env.action_space
                if train_env.action_space.shape:
                    # This is a bit complicated, but it's needed because the last batch
                    # might have a different batch dimension than the env's action
                    # space, (only happens on the last batch in supervised learning).
                    # TODO: Should we perhaps drop the last batch?
                    action_space = train_env.action_space
                    batch_size = getattr(train_env, "num_envs", getattr(train_env, "batch_size", 0))
                    env_is_batched = batch_size is not None and batch_size >= 1
                    if env_is_batched:
                        # NOTE: Need to pass an action space that actually reflects the batch
                        # size, even for the last batch!
                        obs_batch_size = observations.x.shape[0] if observations.x.shape else None
                        action_space_batch_size = (
                            train_env.action_space.shape[0]
                            if train_env.action_space.shape
                            else None
                        )
                        if (
                            obs_batch_size is not None
                            and obs_batch_size != action_space_batch_size
                        ):
                            action_space = batch_space(
                                train_env.single_action_space, obs_batch_size
                            )

                y_preds = action_space.sample()
                rewards = train_env.send(Actions(y_pred=y_preds))

    def get_actions(self, observations: Observations, action_space: Space) -> Actions:
        # This won't work on weirder spaces.
        if action_space.shape:
            assert observations.x.shape[0] == action_space.shape[0]
        if getattr(observations.x, "shape", None):
            batch_size = 1
            if observations.x.ndim > 1:
                batch_size = observations.x.shape[0]
            self.batch_sizes.append(batch_size)
        else:
            self.batch_sizes.append(0)  # X isn't batched.
        return action_space.sample()
