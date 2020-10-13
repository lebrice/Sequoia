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

from common.gym_wrappers import AsyncVectorEnv, EnvDataset
from common.gym_wrappers.utils import has_wrapper
from settings.active.active_dataloader import ActiveDataLoader
from utils.logging_utils import get_logger

from .make_env import make_batched_env
from common.batch import Batch

logger = get_logger(__file__)
T = TypeVar("T")

from common.gym_wrappers.env_dataset import StepResult
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
    
    TODO: Clean up this constructor, it has way too many arguments.
    """
    def __init__(self,
                 env: Union[str, Callable[[], Env]],
                 batch_size: int = 1,
                 max_steps: int = 1_000_000,
                 #   
                 #  observation_space: gym.Space = None,
                 #  action_space: gym.Space = None,
                 #  reward_space: gym.Space = None,
                 #
                 policy: Callable[[ObservationType, gym.Space], ActionType] = None,
                 dataset_item_type: Callable = None,
                 pre_batch_wrappers: List[Callable] = None,
                 post_batch_wrappers: List[Callable] = None,
                 # todo: Use those ?
                 observations_type: Type[Observations] = None,
                #  rewards_type: Type[Rewards] = None,
                #  actions_type: Type[Actions] = None,
                 # TODO: Still debugging both multiprocessing and non-multiprocessing versions. 
                 use_multiprocessing: bool = True,
                 **kwargs):

        if not isinstance(env, str) or callable(env):
            raise RuntimeError(
                f"`env` must be either a str (gym ID) or acallable which takes "
                f"no argument and returns a `gym.Env` (Received {env})."
            )
        self.env_name: Optional[str] = env if isinstance(env, str) else None
        self.max_steps = max_steps
        self.n_parallel_envs: int = batch_size
        # self.batch_transform = batch_transform
        self.observations_type: Type[ObservationType] = observations_type
        # self.rewards_type: Type[RewardType] = rewards_type
        # self.actions_type: Type[ActionType] = actions_type
        
        # NOTE: Since we're gonna pass 'batch_size = None' to the DataLoader
        # constructor below, accessing `self.batch_size` will always return None
        # in the future.
        self._batch_size = batch_size
        # TODO: Move the Policy stuff into a wrapper?
        # self.policy: Callable[[Tensor], Tensor] = policy
        assert not has_wrapper(env, VectorEnv), "Env shouldn't already be vectorized!"
        
        if use_multiprocessing and batch_size > 32:
            # TODO: Maybe add some kind of 'internal' and 'external' batch size?
            # Like, external_batch_size would be the size of the batches returned
            # by this loader, while 'internal_batch_size' would be the number of
            # envs which actually produce data?
            # TODO: Create some kind of hybrid of AsyncVectorEnv and SyncVectorEnv,
            # where there could be more than one env per process, since most of
            # the overhead is probably due to MP, not to the env.
            raise RuntimeError(
                f"TODO: The batch_size arg passed to {__class__} ({batch_size}) "
                f"is too large for the current RL setup, because we currently "
                f"have to create `batch_size` environments, with one process "
                f"per environment. (There are {mp.cpu_count()} CPUs on this "
                f"machine)\n"
                f"If you want larger batches, you could use this loader to "
                f"fill up some kind of replay buffer, and then create batches "
                f"by sampling from that buffer. "
            )
        self.env = make_batched_env(
            env,
            batch_size=batch_size,
            wrappers=pre_batch_wrappers,
            asynchronous=use_multiprocessing,
        )
        self.post_batch_wrappers = post_batch_wrappers or []
        for wrapper in self.post_batch_wrappers:
            self.env = wrapper(self.env)

        from common.gym_wrappers import TransformObservation
        
        # Add a wrapper to create an IterableDataset from the env.
        self.env = EnvDataset(
            self.env,
            policy=policy,
            max_steps=max_steps,
            dataset_item_type=dataset_item_type,
        )

        # We set this to 0 because we are using workers internally in the
        # VecEnv instead of the usual dataloadering workers.
        kwargs["num_workers"] = 0

        super().__init__(
            dataset=self.env,
            # The batch size is None, because the VecEnv takes care of
            # doing the batching for us.
            batch_size=None,
            **kwargs,
        )
        Wrapper.__init__(self, env=self.env)
        self.env: Union[AsyncVectorEnv, EnvDataset]
        self.observation_space: gym.Space = self.env.observation_space
        self.action_space: gym.Space = self.env.action_space
        self.reward_space: gym.Space = spaces.Tuple([
            spaces.Box(
                low=self.env.reward_range[0],
                high=self.env.reward_range[1],
                shape=())
            for _ in range(self.n_parallel_envs)
        ])
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
        return super().__iter__()
        

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
    
    def __del__(self):
        self.env.close()