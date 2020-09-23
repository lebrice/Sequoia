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
                    Sequence, Tuple, Type, TypeVar, Union)

import gym
# import matplotlib.pyplot as plt



from gym import Env, Wrapper


from torch import Tensor
from common.gym_wrappers import AsyncVectorEnv, EnvDataset
from common.gym_wrappers.utils import wrapper_is_present
from settings.active.active_dataloader import ActiveDataLoader
from utils.logging_utils import get_logger

from .make_env import make_batched_env

# from .gym_dataset import GymDataset
# from .zip_dataset import ZipDataset

logger = get_logger(__file__)
EnvFactory = Callable[[], Env]
T = TypeVar("T")

def collate_fn(batches):
    assert False, batches

_UNUSED = "UNUSED"
class GymDataLoader(ActiveDataLoader[Tensor, Tensor, Tensor], gym.Wrapper):
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
                 env: Union[str, Callable[[], Env]],
                 batch_size: int = 1,
                 max_steps: int = 1_000_000,
                 pre_batch_wrappers: List[Union[Type[Wrapper], Tuple[Type[Wrapper], Dict]]] = None,
                 num_workers: int = 0, # unused, forced to 0.
                 **kwargs):
        self.base_env = env
        self.max_steps = max_steps
        self.kwargs = kwargs
        self._batch_size = batch_size
        # TODO: Move the Policy stuff into a wrapper?
        # self.policy: Callable[[Tensor], Tensor] = policy
        self.env: AsyncVectorEnv = make_batched_env(
            env,
            batch_size=batch_size,
            wrappers=pre_batch_wrappers,
        )
        if not isinstance(self.env, EnvDataset):
            # Add a wrapper to create an IterableDataset from the env.
            self.env = EnvDataset(self.env, max_steps=max_steps)

        super().__init__(
            dataset=self.env,
            # We set this to 0 because we are using workers internally in the
            # VecEnv instead of the usual dataloadering workers. 
            num_workers=0,
            # The batch size is None, because the VecEnv takes care of
            # doing the batching for us.
            batch_size=None,
            **kwargs,
        )
        Wrapper.__init__(self, env=self.env)
