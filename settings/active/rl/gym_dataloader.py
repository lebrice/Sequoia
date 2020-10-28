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
from gym.vector.utils import batch_space
from torch import Tensor
from torch.utils.data import IterableDataset


from common.batch import Batch
from common.gym_wrappers.utils import StepResult, has_wrapper
from settings.active.active_dataloader import ActiveDataLoader
from utils.logging_utils import get_logger
from common.gym_wrappers import EnvDataset

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
                 env: Union[EnvDataset, "PolicyEnv"] = None,
                 dataset: Union[EnvDataset, "PolicyEnv"] = None,
                 batch_size: int = None,
                 num_workers: int = None,
                 **kwargs):
        assert not (env is None and dataset is None), "One of the `dataset` or `env` arguments must be passed."
        assert not (env is not None and dataset is not None), "Only one of the `dataset` and `env` arguments can be used."
        
        if not isinstance(env, IterableDataset):
            raise RuntimeError(
                f"The env {env} isn't an interable dataset! (You can use the "
                f"EnvDataset or PolicyEnv wrappers to convert a gym environment "
                "to an IterableDataset)."
            )

        if batch_size:
            assert batch_size > 0, "Batch size should be positive."
            if not isinstance(env.unwrapped, VectorEnv):
                raise RuntimeError(f"The environment should already be a vectorized environment if batch_size is given.")
            if batch_size != env.unwrapped.num_envs:    
                logger.warning(UserWarning(
                    f"The provided batch size {batch_size} will be ignored, since "
                    f"the provided env is vectorized with a batch_size of "
                    f"{env.unwrapped.num_envs}."
                ))
        self.env = env
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
            )
            if isinstance(self.env.unwrapped, VectorEnv):                
                # Same here, we use a 'batched' space rather than Tuple.
                self.reward_space = batch_space(self.reward_space, batch_size)
        self._iterator: Iterator = None

    # def __next__(self) -> EnvDatasetItem:
    #     if self._iterator is None:
    #         self._iterator = self.__iter__()
    #     return next(self._iterator)

    def __len__(self):
        if isinstance(self.env.unwrapped, VectorEnv) and self.max_steps_per_epoch:
            return self.max_steps_per_epoch
        else:
            raise NotImplementedError(f"TODO: Can't tell the length of the env {self.env}.")
    

    def __iter__(self) -> Iterable[ObservationType]:
        
        assert self.num_workers == 0, "Shouldn't be using multiple DataLoader workers!"
        # This would give back a single-process dataloader iterator over the
        # 'dataset' which in this case is the environment:
        # return super().__iter__()
        
        # This, on the other hand, completely bypasses the dataloader iterator,
        # and instead just yields the samples from the dataset directly, which
        # is actually what we want! 
        self._iterator = iter(self.env)
        yield from self._iterator

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
    
    
    
    @classmethod
    def for_env_id(cls,
                   env: str,
                   batch_size: Optional[int],
                   num_workers: int = None,
                   max_epochs: int = None,
                   **kwargs) -> "GymDataLoader":
        """ TODO: The constructor was getting really ugly, because it had to
        have the arguments needed to make the batched environment, as well as
        the EnvDataset, etc. Therefore I'm moving the stuff required for
        creating a Batched EnvDataset here.
        """ 
        assert isinstance(env, str), "Can only call this on environment ids (strings)"
    
        policy: Optional[Callable] = kwargs.pop("policy", None)
        if num_workers is None:
            num_workers = mp.cpu_count()
        logger.info(f"Creating a vectorized version of {env} with batch size of {batch_size} and {num_workers} processes.")
        
        # Make the env / VectorEnv.
        if batch_size:
            env = make_batched_env(base_env=env, batch_size=batch_size, n_workers=num_workers, **kwargs)
        else:
            env = gym.make(env, **kwargs)

        # Make the env iterable.
        if policy:
            from common.gym_wrappers.policy_env import PolicyEnv
            env = PolicyEnv(env, policy=policy)
        else:
            env = EnvDataset(env, max_episodes=max_epochs)
        return cls(env, **kwargs)
