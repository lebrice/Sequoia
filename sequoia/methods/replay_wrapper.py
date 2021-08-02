from pytorch_lightning import Callback, LightningModule, Trainer
from sequoia.settings import Setting
from sequoia.settings.sl import SLSetting, SLEnvironment
from sequoia.settings.sl.continual import Environment
from sequoia.settings.sl.continual.objects import (
    Observations,
    Actions,
    Rewards,
    ObservationType,
    ActionType,
    RewardType,
    ObservationSpace,
)
from torch import nn, Tensor
import torch
from typing import Tuple, Dict, Type, Optional, Iterator
import numpy as np
from sequoia.methods.models.base_model import BaseModel
from sequoia.common.spaces import TypedDictSpace
from sequoia.methods.experience_replay import Buffer
from gym.vector.utils.spaces import batch_space


from sequoia.common.gym_wrappers import IterableWrapper


class ReplayEnvWrapper(IterableWrapper[Environment[ObservationType, ActionType, RewardType]]):
    def __init__(
        self,
        env: Environment[ObservationType, ActionType, RewardType],
        buffer: Buffer = None,
        capacity: int = None,
        sample_size: int = 32,
        task_id: int = None,
    ):
        # NOTE: Passing `call_hooks` of `True` to `IterableWrapper` so that
        # `super().reset()` and `super().step()` use our `self.observation`,
        # `self.action`, `self.reward`, etc hooks.
        super().__init__(env, call_hooks=True)
        if not self.is_vectorized:
            raise RuntimeError(
                f"Can only use the ReplayEnvWrapper on a vectorized env, not {env}."
            )
        self.capacity = capacity
        self.sample_size = sample_size
        self.task_id: Optional[int] = task_id

        if buffer is None:
            assert capacity is not None, "Capacity must be given if `buffer` isn't!"
            extra_buffers = {}
            if "task_labels" in env.observation_space.keys():
                extra_buffers["task_labels"] = torch.LongTensor
            buffer = Buffer(
                capacity=capacity,
                input_shape=env.single_observation_space.x.shape,
                extra_buffers=extra_buffers,
            )
        self.buffer = buffer
        self.capacity = buffer.capacity

        # Wether to augment the observations and rewards with buffer samples.
        self._sampling_enabled = False
        # Wether to store some samples from the environment into the buffer.
        self._collection_enabled = False
        # Items that come from the buffer.
        self._buffer_observation: Optional[Observations] = None
        self._buffer_reward: Optional[Rewards] = None
        self._buffer_action: Optional[Actions] = None
        # Items that come from the wrapped environment.
        self._env_observation: Optional[Observations] = None
        self._env_reward: Optional[Rewards] = None
        self._env_action: Optional[Actions] = None

        self._epochs = -1

    def sample(self, n: int = None) -> Tuple[ObservationType, RewardType]:
        samples = self.buffer.sample(n or self.sample_size, exclude_task=self.task_id)

        obs_kwargs = {"x": samples["x"]}
        if "task_labels" in samples:
            obs_kwargs.update(task_labels=samples["task_labels"])
        obs = self.observation_space.dtype(**obs_kwargs)

        reward_kwargs = {"y": samples["y"]}
        # TODO: Once we also use a TypedDictSpace for the Actions/Rewards, do this:
        # rewards = self.reward_space.dtype(**reward_kwargs)
        rewards = Rewards(**reward_kwargs)
        return obs, rewards

    def add_reservoir(self, observation: ObservationType, reward: RewardType) -> None:
        values = {"x": observation.x}
        if "task_labels" in observation:
            values.update(task_labels=observation.task_labels)
        values.update(y=reward.y)
        self.buffer.add_reservoir(values)

    def reset(self) -> ObservationType:
        self._epochs += 1
        # NOTE: Since `reset` is the beginning of an epoch in SL, we disable collection
        # on the start of the second epoch.
        if self.collection_enabled and self._epochs == 1:
            self.disable_collection()

        obs = super().reset()
        # obs = self.observation(obs)
        return obs

    def send(self, action: ActionType) -> RewardType:
        return self._iterator.send(action)

        # return super().send(action)
        # action = self.action(action)
        # reward = self.env.send(action)
        # if self._reward_applied:
        #     return reward
        # return self.reward(reward)
        # NOTE: Could use this if we opt for the 'generator-style' below:
        # self.action_ = action

    @classmethod
    def iterator(cls, env: "ReplayEnvWrapper") -> Iterator:
        # This would be cool, but we'd need to somehow store the iterator somewhere.
        obs = env.reset()
        done = False
        while not done:
            action = yield obs, None

            if action is None:
                action = env.action_
            assert action is not None

            obs, reward, done, info = env.step(action)
            yield reward
    
    def __iter__(self) -> Iterator:
        # TODO: This would work, if we were able to send the actions to the iterator.
        # obs = self.reset()
        self._iterator = self.iterator(self)
        yield from self._iterator

        # for obs, rewards in self.env:
        #     obs = self.observation(obs)
        #     if rewards is not None:
        #         self._reward_applied = True
        #         rewards = self.reward(rewards)
        #     yield obs, rewards

    def step(self, action: ActionType) -> Tuple[ObservationType, RewardType, bool, dict]:
        # NOTE: Don't need to call these hooks, since the `IterableWrapper` does it for
        # us.
        assert self.call_hooks
        # action = self.action(action)
        obs, rewards, done, info = super().step(action)
        print(f"Step: {rewards.y}")
        # obs = self.observation(obs)
        # rewards = self.reward(rewards)
        # done = done
        # info = info
        return obs, rewards, done, info

    def observation(self, observation: ObservationType) -> ObservationType:
        """ Augments the observations with the samples from the buffer.
        """
        if self.collection_enabled:
            # push these samples into the buffer?
            self._env_observation = observation

        if self.sampling_enabled:
            self._buffer_observation, self._buffer_reward = self.sample()
            return type(observation).concatenate([observation, self._buffer_observation])
        return observation

    def reward(self, reward: RewardType) -> RewardType:
        """ Augments the rewards with the samples from the buffer.
        """
        if self.collection_enabled:
            # push these samples into the buffer?
            self._env_reward = reward
            assert self._env_observation is not None
            self.add_reservoir(self._env_observation, self._env_reward)
            self._env_observation = None
            self._env_reward = None

        if self.sampling_enabled:
            return type(reward).concatenate([reward, self._buffer_reward])
        return reward

    def action(self, action: ActionType) -> ActionType:
        """ Splits off the action into the ones that are meant for the buffer and those
        meant for the wrapped environment.
        """
        if self.collection_enabled:
            # In the case of RL, we'd need to store the actions as well.
            pass
        if self.sampling_enabled:
            self._buffer_action = action[:, self.sample_size :]
            self._env_action = action[:, : self.sample_size]
            return self._env_action
        return action

    @property
    def sampling_enabled(self) -> bool:
        return self._sampling_enabled

    def enable_sampling(self) -> None:
        self._sampling_enabled = True
        n = self.env.batch_size + self.sample_size
        self.observation_space = batch_space(self.single_observation_space, n=n)
        self.action_space = batch_space(self.single_action_space, n=n)
        self.reward_space = batch_space(self.single_reward_space, n=n)

    def disable_sampling(self) -> None:
        self._sampling_enabled = False
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_space = self.env.reward_space

    @property
    def collection_enabled(self) -> bool:
        return self._collection_enabled

    def enable_collection(self) -> None:
        self._collection_enabled = True

    def disable_collection(self) -> None:
        self._collection_enabled = False

    def for_next_env(
        self, other_env: Environment[ObservationType, ActionType, RewardType], task_id: int = None
    ) -> "ReplayEnvWrapper[Environment[ObservationType, ActionType, RewardType]]":
        """ Creates a new ReplayEnvWrapper around `other_env` and bootstraps it with the
        state of `self`.
        
        Also sets up the next env so that it will draw samples from the buffer.
        """
        wrapper = type(self)(
            other_env,
            buffer=self.buffer,
            capacity=self.capacity,
            sample_size=self.sample_size,
            task_id=task_id,
        )
        if task_id != self.task_id:
            wrapper.enable_sampling()
        if self._collection_enabled:
            wrapper.enable_collection()
        return wrapper

