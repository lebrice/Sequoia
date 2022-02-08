""" Dataloader for a Gym Environment. Uses multiple parallel environments.

TODO: @lebrice: We need to decide which of these two behaviours we want to
    support in the GymDataLoader, (if not both):

- Either iterate over the dataset and get the usual 4-item tuples like gym,
    by using a policy to generate the actions,
OR
- Give back 3-item tuples (without the reward) and give the reward when
    users send back an action for the current observation. Users would either
    be required to send actions back after each observation or to provide a
    policy to "fill-in-the-gaps" and select the action when the model doesn't
    send one back.

The traditional supervised dataloader can be easily recovered in this second
case: since the reward doesn't depend on the action, we can just send back a
random or None action to the dataloader, and group the returned reward with
the batch of observations, before yielding the (observations, rewards)
batch.

In either case, we can easily keep the `step` API from gym available.
Need to talk more about this for sure.
"""
import warnings
from typing import Any, Iterable, Iterator, Optional, TypeVar, Union

import gym
import numpy as np
from gym import Wrapper, spaces
from gym.utils.colorize import colorize
from gym.vector import AsyncVectorEnv, VectorEnv
from gym.vector.utils import batch_space
from torch import Tensor
from torch.utils.data import IterableDataset

from sequoia.common.gym_wrappers import EnvDataset, IterableWrapper
from sequoia.common.gym_wrappers.policy_env import PolicyEnv
from sequoia.common.gym_wrappers.utils import StepResult
from sequoia.settings.base.objects import Actions
from sequoia.settings.rl.environment import ActiveEnvironment
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")


# TODO: The typing information from sequoia.settings.base.environment isn't quite
# accurate here... The observations are bound by Tensors or numpy arrays, not
# 'Batch' objects.

# from sequoia.settings.base.environment import ObservationType, ActionType, RewardType
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")


class GymDataLoader(
    ActiveEnvironment[ObservationType, ActionType, RewardType], IterableWrapper, Iterable
):
    """Environment for RL settings.

    Exposes **both** the `gym.Env` as well as the "Active" DataLoader APIs.

    This is useful because it makes it easy to adapt a method originally made for SL so
    that it can also work in a reinforcement learning context, where the rewards (e.g.
    image labels, or correct/incorrect prediction, etc.) are only given *after* the
    action (e.g. y_pred) has been received by the environment.

    meaning you
    can use this in two different ways:

    1. Gym-style using `step`:
        1. Agent   --------- action ----------------> Env
        2. Agent   <---(state, reward, done, info)--- Env

    2. ActiveDataLoader style, using `iter` and `send`:
        1. Agent   <--- (state, done, info) --- Env
        2. Agent   ---------- action ---------> Env
        3. Agent   <--------- reward ---------- Env


    This would look something like this in code:

    ```python
    env = GymDataLoader("CartPole-v0", batch_size=32)
    for states, done, infos in env:
        actions = actor(states)
        rewards = env.send(actions)
        loss = loss_function(...)

    # OR:

    state = env.reset()
    for i in range(max_steps):
        action = self.actor(state)
        states, reward, done, info = env.step(action)
        loss = loss_function(...)
    ```

    """

    def __init__(
        self,
        env: Union[EnvDataset, PolicyEnv] = None,
        dataset: Union[EnvDataset, PolicyEnv] = None,
        batch_size: int = None,
        num_workers: int = None,
        **kwargs,
    ):
        assert not (
            env is None and dataset is None
        ), "One of the `dataset` or `env` arguments must be passed."
        assert not (
            env is not None and dataset is not None
        ), "Only one of the `dataset` and `env` arguments can be used."

        if not isinstance(env, IterableDataset):
            raise RuntimeError(
                f"The env {env} isn't an interable dataset! (You can use the "
                f"EnvDataset or PolicyEnv wrappers to make an IterableDataset "
                f"from a gym environment."
            )

        if isinstance(env.unwrapped, VectorEnv):
            if batch_size is not None and batch_size != env.num_envs:
                logger.warning(
                    UserWarning(
                        f"The provided batch size {batch_size} will be ignored, since "
                        f"the provided env is vectorized with a batch_size of "
                        f"{env.unwrapped.num_envs}."
                    )
                )
            batch_size = env.num_envs

        if isinstance(env.unwrapped, AsyncVectorEnv):
            num_workers = env.num_envs
        else:
            num_workers = 0

        self.env = env
        # NOTE: The batch_size and num_workers attributes reflect the values from the
        # iterator (the VectorEnv), not those of the dataloader.
        # This is done in order to avoid pytorch workers being ever created, and also so
        # that pytorch-lightning stops warning us that the num_workers is too low.
        self._batch_size = batch_size
        self._num_workers = num_workers
        super().__init__(
            dataset=self.env,
            # The batch size is None, because the VecEnv takes care of
            # doing the batching for us.
            batch_size=None,
            num_workers=0,
            collate_fn=None,
            **kwargs,
        )
        Wrapper.__init__(self, env=self.env)
        assert not isinstance(self.env, GymDataLoader), "Something very wrong is happening."
        # self.max_epochs: int = max_epochs
        self.observation_space: gym.Space = self.env.observation_space
        self.action_space: gym.Space = self.env.action_space
        self.reward_space: gym.Space
        if isinstance(env.unwrapped, VectorEnv):
            env: VectorEnv
            batch_size = env.num_envs
            # TODO: Overwriting the action space to be the 'batched' version of
            # the single action space, rather than a Tuple(Discrete, ...) as is
            # done in the gym.vector.VectorEnv.
            self.action_space = batch_space(env.single_action_space, batch_size)

        if not hasattr(self.env, "reward_space"):
            self.reward_space = spaces.Box(
                low=self.env.reward_range[0],
                high=self.env.reward_range[1],
                shape=(),
                dtype=np.float64,
            )
            if isinstance(self.env.unwrapped, VectorEnv):
                # Same here, we use a 'batched' space rather than Tuple.
                self.reward_space = batch_space(self.reward_space, batch_size)

        # BUG: Fix this bug: the observation / action spaces don't accept Tensors as
        # valid samples, even though they should.
        # self.observation_space = add_tensor_support(self.observation_space)
        # self.action_space = add_tensor_support(self.action_space)
        # self.reward_space = add_tensor_support(self.reward_space)
        # assert has_tensor_support(self.observation_space)

    @property
    def num_workers(self) -> Optional[int]:
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value: Any) -> Optional[int]:
        if value and value != self._num_workers:
            warnings.warn(
                RuntimeWarning(
                    f"Can't set num_workers to {value}, it's hard-set to {self._num_workers}"
                )
            )

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: Any) -> Optional[int]:
        if value != self._batch_size:
            warnings.warn(
                RuntimeWarning(
                    f"Can't set batch size to {value}, it's hard-set to {self._batch_size}"
                )
            )

    def __next__(self) -> ObservationType:
        if self._iterator is None:
            self._iterator = self.__iter__()
        return next(self._iterator)

    # def __len__(self):
    #     if isinstance(self.env, EnvDataset):
    #         return self.env.max_steps
    #     raise NotImplementedError(f"TODO: Can't tell the length of the env {self.env}.")

    def _obs_have_done_signal(self) -> bool:
        """Try to determine if the observations contain the 'done' signal or not."""
        if (
            isinstance(self.observation_space, spaces.Dict)
            and "done" in self.observation_space.spaces
        ):
            return True
        return False

    def __iter__(self) -> Iterator:
        # TODO: Pretty sure this could be greatly simplified by just always using the loop from EnvDataset.
        # return super().__iter__()
        # assert False, self.env.__iter__()
        if self.is_vectorized:
            # elif isinstance(self.observation_space, spaces.Tuple)
            if not self._obs_have_done_signal():
                warnings.warn(
                    RuntimeWarning(
                        colorize(
                            f"You are iterating over a vectorized env, but the observations "
                            f"don't seem to contain the 'done' signal! You should definitely "
                            f"consider applying something like an `AddDoneToObservation` "
                            f"wrapper to each individual env before vectorization. ",
                            "red",
                        )
                    )
                )
        return self.env.__iter__()
        # yield from IterableWrapper.__iter__(self)

        # self.observation_ = self.reset()
        # self.done_ = False
        # self.action_ = None
        # self.reward_ = None

        # # Yield the first observation_.
        # # TODO: Maybe add something like 't' on the observations to make sure they
        # # line up with the rewards we get?
        # yield self.observation_

        # if self.action_ is None:
        #     raise RuntimeError(
        #         f"You have to send an action using send() between every "
        #         f"observation. (env = {self})"
        #     )
        # def done_is_true(done: Union[bool, np.ndarray, Sequence[bool]]) -> bool:
        #     return done if isinstance(done, bool) or not done.shape else all(done)

        # while not any([done_is_true(self.done_), self.is_closed()]):
        #     # logger.debug(f"step {self.n_steps_}/{self.max_steps},  (episode {self.n_episodes_})")

        #     # Set those to None to force the user to call .send()
        #     self.action_ = None
        #     self.reward_ = None
        #     yield self.observation_

        #     if self.action_ is None:
        #         raise RuntimeError(
        #             f"You have to send an action using send() between every "
        #             f"observation. (env = {self})"
        #         )

    # def __iter__(self) -> Iterable[ObservationType]:
    #     # This would give back a single-process dataloader iterator over the
    #     # 'dataset' which in this case is the environment:
    #     # return super().__iter__()

    #     # This, on the other hand, completely bypasses the dataloader iterator,
    #     # and instead just yields the samples from the dataset directly, which
    #     # is actually what we want!
    #     # BUG: Somehow this doesn't batch the samples correctly..
    #     return self.env.__iter__()

    #     # TODO: BUG: Wrappers applied on top of the GymDataLoader won't have an
    #     # effect on the values yielded by this iterator. Currently trying to fix
    #     # this inside the IterableWrapper base class, but it's not that simple.

    #     # return type(self.env).__iter__(self)
    #     # if has_wrapper(self.env, EnvDataset):
    #     #     return EnvDataset.__iter__(self)
    #     # elif has_wrapper(self.env, PolicyEnv):
    #     #     return PolicyEnv.__iter__(self)
    #     # return type(self.env).__iter__(self)
    #     # return  iter(self.env)
    #     # yield from self._iterator

    #     # Could increment the number of epochs here also, if we wanted to keep
    #     # count.

    # def random_actions(self):
    #     return self.env.random_actions()

    def step(self, action: Union[ActionType, Any]) -> StepResult:
        # logger.debug(f"Calling step on self.env")
        return super().step(action)

    def send(self, action: Union[ActionType, Any]) -> RewardType:
        # TODO: Remove this unwrapping code, and instead only unwrap stuff if necessary
        # for the environment.
        if isinstance(action, Actions):
            action = action.y_pred
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray) and not action.shape:
            action = action.item()
        if isinstance(self.env.action_space, spaces.Tuple) and isinstance(action, np.ndarray):
            action = action.tolist()
        assert action in self.env.action_space, (action, self.env.action_space)
        return super().send(action)
        # self.action_ = action
        # self.observation_, self.reward_, self.done_, self.info_ = su(action)
        # return self.reward_
        # return self.env.send(action)
