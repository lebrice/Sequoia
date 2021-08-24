from abc import ABC
from typing import Generic, Iterator, Optional, Sequence, TypeVar, Union

import gym
import numpy as np
from gym.vector import VectorEnv
from torch.utils.data import IterableDataset
from sequoia.utils.logging_utils import get_logger
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.utils import MayCloseEarly, has_wrapper, is_proxy_to

logger = get_logger(__file__)

EnvType = TypeVar("EnvType", bound=gym.Env)
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")


class IterableWrapper(MayCloseEarly, IterableDataset, Generic[EnvType], ABC):
    """ABC for a gym Wrapper that supports iterating over the environment.

    This allows us to wrap dataloader-based Environments and still use the gym
    wrapper conventions, as well as iterate over a gym environment as in the
    Active-dataloader case.

    NOTE: We have IterableDataset as a base class here so that we can pass a wrapped env
    to the DataLoader function. This wrapper however doesn't perform the actual
    iteration, and instead depends on the wrapped environment already supporting
    iteration.
    """

    def __init__(self, env: gym.Env, call_hooks: bool = False):
        super().__init__(env)
        from sequoia.settings.sl import PassiveEnvironment

        self.wrapping_passive_env = isinstance(self.unwrapped, PassiveEnvironment)
        # Wether we should automatically apply the `hooks` in the `step` and `reset`.
        self.call_hooks = call_hooks
        # Flag used to tell if the `reward` method has already been applied to the given
        # batch.
        self._reward_applied = False

    @property
    def is_vectorized(self) -> bool:
        """Returns wether this wrapper is wrapping a vectorized environment."""
        return isinstance(self.unwrapped, VectorEnv)

    def __next__(self):
        # TODO: This is tricky. We want the wrapped env to use *our* step,
        # reset(), action(), observation(), reward() methods, instead of its own!
        # Otherwise if we are transforming observations for example, those won't
        # be affected.
        # logger.debug(f"Wrapped env {self.env} isnt a PolicyEnv or an EnvDataset")
        # return type(self.env).__next__(self)
        from sequoia.settings.rl.environment import ActiveDataLoader

        # from sequoia.settings.sl.environment import PassiveEnvironment

        if has_wrapper(self.env, EnvDataset) or is_proxy_to(
            self.env, (EnvDataset, ActiveDataLoader)
        ):
            obs, reward, done, info = self.step(self.action_)
            return obs
            # raise RuntimeError(f"WIP: Dropping this '__next__' API in RL.")
            # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__iter__.")
            # return EnvDataset.__next__(self)
            # return EnvDataset.__next__(self)
        return self.env.__next__()
        # return self.observation(obs)

    def observation(self, observation):
        # logger.debug(f"Observation won't be transformed.")
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    def done(self, done):
        return done

    def info(self, info):
        return info

    # TODO: Not sure if we should use all the hooks here. Instead, should probably leave
    # it to the wrapper to do.
    def reset(self) -> ObservationType:
        obs = super().reset()
        if self.call_hooks:
            assert callable(self.observation), self.observation
            obs = self.observation(obs)
        return obs

    def step(self, action):
        if self.call_hooks:
            action = self.action(action)
        obs, rewards, done, info = super().step(action)
        if self.call_hooks:
            obs = self.observation(obs)
            rewards = self.reward(rewards)
            done = self.done(done)
            info = self.info(info)
        return obs, rewards, done, info

    # def __len__(self):
    #     return self.env.__len__()

    def length(self) -> Optional[int]:
        """Attempts to return the "length" (in number of `step` calls) of this env.

        When not possible, returns None.

        NOTE: This is a bit ugly, but the idea seems alright.
        """
        try:
            # Try to call self.__len__() without recursing into the wrapped env:
            return len(self)
        except TypeError:
            pass
        try:
            # Try to call self.env.__len__() without recursing into the wrapped^2 env:
            return len(self.env)
        except TypeError:
            pass
        try:
            # Try to call self.env.__len__(), allowing recursing down the chain:
            return self.env.__len__()
        except AttributeError:
            pass
        try:
            # If all else fails, delegate to the wrapped env's length() method, if any:
            return self.env.length()
        except AttributeError:
            pass
        # In the worst case, return None, meaning that we don't have a length.
        return None

    def send(self, action):
        # TODO: Make `send` use `self.step`, that way wrappers can apply the same way to
        # RL and SL environments.
        if self.wrapping_passive_env:
            action = self.action(action)
            reward = self.env.send(action)
            reward = self.reward(reward)
            return reward

            # if not self._reward_applied:
            #     reward = self.reward(reward)
            #     self._reward_applied = True

            # return reward

        self.action_ = action
        self.unwrapped.action_ = action
        self.observation_, self.reward_, self.done_, self.info_ = self.step(action)
        return self.reward_


    def __iter__(self) -> Iterator:
        # TODO: Pretty sure this could be greatly simplified by just always using the
        # loop from EnvDataset.
        if self.wrapping_passive_env:
            # NOTE: Also applies the `self.observation` `self.reward` methods while
            # iterating.
            for obs, rewards in self.env:
                self._reward_applied = False
                obs = self.observation(obs)

                if rewards is not None:
                    assert not self._reward_applied
                    rewards = self.reward(rewards)
                    self._reward_applied = True
                    self._reward = rewards

                yield obs, rewards
        else:
            self.observation_ = self.reset()
            self.done_ = False
            self.action_ = None
            self.reward_ = None

            # Yield the first observation_.
            yield self.observation_

            if self.action_ is None:
                raise RuntimeError(
                    f"You have to send an action using send() between every "
                    f"observation. (env = {self})"
                )

            def done_is_true(done: Union[bool, np.ndarray, Sequence[bool]]) -> bool:
                return done if isinstance(done, bool) or not done.shape else all(done)

            while not any([done_is_true(self.done_), self.is_closed()]):
                logger.debug(
                    f"step {self.n_steps_}/{self.max_steps}, "
                    f"(episode {self.n_episodes_})"
                )

                # Set those to None to force the user to call .send()
                self.action_ = None
                self.reward_ = None
                yield self.observation_

                if self.action_ is None:
                    raise RuntimeError(
                        f"You have to send an action using send() between every "
                        f"observation. (env = {self})"
                    )


class SimplerIterableWrapper(MayCloseEarly, IterableDataset, Generic[EnvType], ABC):
    """ WIP: Simpler version of IterableWrapper. """
    def __init__(self, env: gym.Env, call_hooks: bool = False):
        super().__init__(env=env)
        self.call_hooks = call_hooks
        from sequoia.settings.sl import PassiveEnvironment
        self.wrapping_passive_env = isinstance(self.unwrapped, PassiveEnvironment)

    @property
    def is_vectorized(self) -> bool:
        """Returns wether this wrapper is wrapping a vectorized environment."""
        return isinstance(self.unwrapped, VectorEnv)

    def reset(self):
        obs = super().reset()
        if self.call_hooks:
            return self.observation(obs)
        return obs

    def step(self, action: ActionType):
        if self.call_hooks:
            action = self.action(action)
        obs, rewards, done, info = super().step(action)
        if self.call_hooks:
            obs = self.observation(obs)
            rewards = self.reward(rewards)
            done = self.done(done)
            info = self.info(info)
        return obs, rewards, done, info

    def observation(self, observation):
        # logger.debug(f"Observation won't be transformed.")
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    def done(self, done):
        return done

    def info(self, info):
        return info

    # TODO: test this out.
    def rl_iterator(self) -> Iterator:
        # This would be cool, but we'd need to somehow store the iterator somewhere.
        obs = self.reset()
        done = False
        while not done:
            action = yield obs
            if action is None:
                raise RuntimeError(
                    "Need to pass an action to the generator using `send` after "
                    "receiving each observation."
                )
                # action = self.action_
            assert action is not None

            obs, reward, done, info = self.step(action)
            yield reward

    def sl_iterator(self) -> Iterator:
        env_iterable = self.env
        for i, (obs, rew) in enumerate(env_iterable):
            if self.call_hooks:
                obs = self.observation(obs)
                if rew is not None:
                    rew = self.reward(rew)
            action = yield obs, rew
            if action is not None:
                if self.call_hooks:
                    action = self.action(action)
                rewards = env_iterable.send(action)
                if self.call_hooks:
                    rewards = self.reward(rewards)
                yield rewards

    def __iter__(self) -> Iterator:
        if self.wrapping_passive_env:
            self._iterator = self.sl_iterator()
        else:
            self._iterator = self.rl_iterator()
        return self._iterator

    def send(self, action: ActionType) -> RewardType:
        if self.call_hooks:
            action = self.action(action)
        # if self._iterator is not None:
        #     rewards = self._iterator.send(action)
        # else:
        rewards = self.env.send(action)
        if self.call_hooks:
            rewards = self.reward(rewards)
        return rewards
