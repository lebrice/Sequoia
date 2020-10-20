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

from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union, Iterator)
import multiprocessing as mp
import gym
from gym import Env, Wrapper, spaces
from gym.vector import VectorEnv
from torch import Tensor
from torch.utils.data import IterableDataset


from common.batch import Batch
from common.gym_wrappers.utils import StepResult
from common.gym_wrappers import AsyncVectorEnv
from common.gym_wrappers.utils import has_wrapper, batch_space
from settings.active.active_dataloader import ActiveDataLoader
from utils.logging_utils import get_logger

from .make_env import make_batched_env

logger = get_logger(__file__)
T = TypeVar("T")

from settings.base.environment import Observations, Actions, Rewards


# TODO: The typing information from settings.base.environment isn't quite
# accurate here... The observations are bound by Tensors or numpy arrays, not
# 'Batch' objects.

# from settings.base.environment import ObservationType, ActionType, RewardType
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")


# class GymDataLoader(ActiveDataLoader[Tensor, Tensor, Tensor], gym.Wrapper):
class GymDataLoader(ActiveDataLoader[ObservationType, ActionType, RewardType], gym.Wrapper, Iterable):
    """[WIP] ActiveDataLoader for batched Gym envs.
    
    Exposes **both** the ActiveDataLoader API as well as the usual `gym.Env`
    API, meaning you can use this in two different ways:
    
    1. ActiveDataLoader style, using `iter` and `send`:
        1. Agent   <--- (state, done, info) --- Env
        2. Agent   ---------- action ---------> Env
        3. Agent   <--------- reward ---------- Env

    2. Gym-style using `step`:
        1. Agent   --------- action ----------------> Env 
        2. Agent   <---(state, reward, done, info)--- Env 

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
    def __init__(self,
                 env: Union[IterableDataset, gym.Env],
                 batch_size: int = None,
                 **kwargs):
        # We set this to 0 because we are using workers internally in the
        # VecEnv instead of the usual dataloadering workers.
        kwargs["num_workers"] = 0
        self.env = env
        super().__init__(
            dataset=self.env,
            # The batch size is None, because the VecEnv takes care of
            # doing the batching for us.
            batch_size=None,
            **kwargs,
        )
        Wrapper.__init__(self, env=self.env)

        self.observation_space: gym.Space = self.env.observation_space
        self.action_space: gym.Space = self.env.action_space        

        if isinstance(env.unwrapped, VectorEnv):
            env: VectorEnv
            batch_size = env.num_envs
            # TODO: Overwriting the action space to be the 'batched' version of
            # the single action space, rather than a Tuple(Discrete, ...).
            self.action_space = batch_space(env.single_action_space, batch_size)

        if not hasattr(self.env, "reward_space"):
            self.reward_space = spaces.Box(
                low=self.env.reward_range[0],
                high=self.env.reward_range[1],
                shape=(),
            )
            # Determine wether the env is a vectored env
            if isinstance(self.env.unwrapped, VectorEnv):                
                # TODO: Same here, we use a 'batched' space rather than Tuple.
                self.reward_space = batch_space(self.reward_space, batch_size)
                # self.reward_space = spaces.Tuple([
                #     self.reward_space
                #     for _ in range(self.env.num_envs)
                # ])
        self._iterator: Iterator = None

    # def __next__(self) -> EnvDatasetItem:
    #     if self._iterator is None:
    #         self._iterator = self.__iter__()
    #     return next(self._iterator)

    def __iter__(self) -> Iterable[ObservationType]:
        # logger.debug(f"Resetting the env")
        # self.reset()
        # logger.debug(f"Done resetting the env.")
        # self.env.reset()
        assert self.num_workers == 0, "Shouldn't be using multiple DataLoader workers!"
        # return self.env
        # This gives back the single-process dataloader iterator over the 'dataset'
        # which in this case is the environment:
        return iter(self.env)
        # return super().__iter__()
        

    def set_policy(self, policy: Callable[[ObservationType], ActionType]) -> None:
        self.env.set_policy(policy)

    def random_actions(self):
        return self.env.random_actions()

    def step(self, action: Union[ActionType, Any]) -> StepResult:
        # logger.debug(f"Calling step on self.env")
        return self.env.step(action)

    def send(self, action: Union[ActionType, Any]) -> RewardType:
        # if self.actions_type and not isinstance(action, self.actions_type):
        #     raise RuntimeError(f"Expected to receive an action of type {self.actions_type}?")
        # logger.debug(f"Receiving actions {action}")
        return self.env.send(action)
    