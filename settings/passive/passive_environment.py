"""TODO: Creates a Gym Environment (and DataLoader) from a traditional
Supervised dataset. 
"""

from typing import *

import gym
import torch
import numpy as np
from gym import spaces
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from common.transforms import Compose
from utils.logging_utils import get_logger

from ..base.environment import (Actions, ActionType, Environment, Observations,
                                ObservationType, Rewards, RewardType)
from torch.utils.data.dataloader import _BaseDataLoaderIter
logger = get_logger(__file__)
from common.batch import Batch
from common.gym_wrappers.utils import space_with_new_shape

class PassiveEnvironment(DataLoader, Environment[Tuple[ObservationType,
                                                       Optional[ActionType]],
                                                 ActionType,
                                                 RewardType]):
    """Environment in which actions have no influence on future observations.

    Can either be iterated on like a normal DataLoader, in which case it gives
    back the observation and the reward at the same time, or as a gym
    Environment, in which case it gives the rewards and the next batch of
    observations once an action is given.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this
    category. Similarly to Environment, this just adds some methods on top of
    the usual PyTorch DataLoader.
    """
    passive: ClassVar[bool] = True
    
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self,
                 dataset: Union[IterableDataset, Dataset],
                 split_batch_fn: Callable[[Tuple[Any, ...]], Tuple[ObservationType, ActionType]] = None,
                 observation_space: gym.Space = None,
                 action_space: gym.Space = None,
                 reward_space: gym.Space = None,
                 n_classes: int = None,
                 adjust_spaces_with_data: bool = True,
                #  pretend_to_be_active: bool = False,
                 **kwargs):
        """Creates the DataLoader/Environment for the given dataset.

        Parameters
        ----------
        dataset : [type]
            The dataset to iterate on. Should ideally be indexable (a Map-style
            dataset).

        TODO: IDK if this is the best way to go about it. Maybe it would be best
        to just adopt the gym Env API and forget about this "send" idea here. 
        pretend_to_be_active : bool, optional
            Wether to withhold the rewards (labels) from the batches when being
            iterated on like the usual dataloader, and to only give them back
            after an action is received through the 'send' method. False by
            default, in which case this behaves exactly as a normal dataloader
            when being iterated on.

        **kwargs:
            The rest of the usual DataLoader kwargs.
        """
        super().__init__(dataset=dataset, **kwargs)
        self.split_batch_fn = split_batch_fn
        
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space
        self.reward_space: gym.Space = reward_space
        self.n_classes: Optional[int] = n_classes

        self._iterator: Optional[_BaseDataLoaderIter] = None
        # NOTE: These here are never processed with self.observation or self.reward. 
        self._previous_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._current_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._next_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._done: Optional[bool] = None
        self._closed: bool = False

        if adjust_spaces_with_data:
            self._adjust_spaces_using_data()

    def _adjust_spaces_using_data(self) -> None:
        """ Adjust the observation / reward spaces to reflect the data, if
        possible.
        """ 
        observation_space = self.observation_space
        action_space = self.action_space
        reward_space = self.reward_space
        n_classes = self.n_classes
        
        # Temporarily create an iterator to get a batch of data.
        temp_iterator = iter(self)
        sample_batch = next(temp_iterator)
        del temp_iterator

        if not isinstance(sample_batch, (tuple, list, Batch)):
            raise RuntimeError(f"Batches should be lists, tuples or Batch "
                               f"objects, not {type(sample_batch)}.")
        
        # if self.split_batch_fn:
        #     # Use the function to split the batch.
        #     sample_batch = self.split_batch_fn(sample_batch)
        if len(sample_batch) != 2:
            raise RuntimeError("Need to pass a split_batch_fn since batches "
                               "don't have length 2.")

        observations, rewards = sample_batch
        # assert isinstance(observations, Observations), "Assuming this for now."
        # assert isinstance(rewards, Rewards), "Assuming this for now."
        
        if isinstance(observations, (Tensor, np.ndarray)):
            observations = (observations,)
        if isinstance(rewards, (Tensor, np.ndarray)):
            rewards = (rewards,)

        
        image_batch = observations[0]
        image = image_batch[0]
        if not observation_space:
            assert image.min() >= 0., "Assuming this for now."
            assert image.max() <= 1., "Assuming this for now."
            assert len(image.shape) >= 3, image.shape
            if len(observations) == 1:
                observation_space = spaces.Box(low=0, high=1, shape=image.shape)
            else:
                raise RuntimeError(f"Can't infer the space when observations have more than one tensor.")
        else:
            # Adjust the obs space to match the shape of the real observations.
            observation_space = space_with_new_shape(
                observation_space,
                image.shape if len(observations) == 1 else
                tuple(tensor.shape[1:] for tensor in observations)
            )

        self.observation_space = spaces.Tuple([
            observation_space for _ in range(self.batch_size)
        ])
        
        reward_batch = rewards[0]
        reward = reward_batch[0]
        if not reward_space:
            if action_space:
                reward_space = action_space
            elif n_classes is not None:
                reward_space = spaces.Discrete(n=n_classes)
            else:
                raise RuntimeError("Need n_classes or action_space when "
                                   "reward_space isn't given.")
        else:
            # Adjust the reward space to match the shape of the actual rewards.
            reward_space = space_with_new_shape(
                reward_space,
                reward.shape if len(rewards) == 1 else
                tuple(tensor.shape[1:] for tensor in rewards)
            )

        self.reward_space = spaces.Tuple([
            reward_space for _ in range(self.batch_size)
        ])

        if not action_space:
            assert reward_space
            action_space = reward_space
        self.action_space = spaces.Tuple([
            action_space for _ in range(self.batch_size)
        ])
        # Batch these spaces to reflect the batch size.
        # TODO: Should we be doing this? This is so we match the AsyncVectorEnv
        # from the gym.vector API.
        # NOTE: On the last batch, if drop_last = False, the observations/actions
        # will not reflect the spaces. Not sure if this could be a problem later.
        # NOTE: Since we set the same object instance at each index, then
        # modifying just one would modify all of them.    
        
        


    def reset(self) -> ObservationType:
        """ Resets the env by deleting and re-creating the iterator.
        Returns the first batch of observations.
        """
        del self._iterator
        self._iterator = self.__iter__()
        self._previous_batch = None
        self._current_batch = self.get_next_batch()
        self._done = False
        # TODO: Not sure if we should be doing this here.
        # self._next_batch = next(self._iterator, None)
        obs = self._current_batch[0]
        return self.observation(obs)

    def close(self) -> None:
        del self._iterator
        self._closed = True

    def get_next_batch(self) -> Tuple[ObservationType, RewardType]:
        """Gets the next batch from the underlying dataset.

        Uses the `split_batch_fn`, if needed. Does NOT apply the self.observation
        and self.reward methods.
        
        Returns
        -------
        Tuple[ObservationType, RewardType]
            [description]
        """
        if self._iterator is None:
            self._iterator = self.__iter__()
        batch = next(self._iterator)
        # if self.split_batch_fn:
        #     batch = self.split_batch_fn(batch)
        return batch
        # obs, reward = batch
        # return self.observation(obs), self.reward(reward)

    def step(self, action: ActionType) -> Tuple[ObservationType, RewardType, bool, Dict]:
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Can't step on a closed env.")
        if self._done is None:
            raise gym.error.ResetNeeded("Need to reset the env before calling step.")
        if self._done:
            raise gym.error.ResetNeeded("Need to reset the env since it is done.")

        # IDEA: Let subclasses customize how the action impacts the env?
        self.use_action(action)
        
        # Transform the Action, if needed:
        action = self.action(action)

        # NOTE: This prev/current/next setup is so we can give the right 'done'
        # signal.
        self._previous_batch = self._current_batch
        if self._next_batch is None:
            # This should only ever happen right after resetting.
            self._next_batch = next(self._iterator, None)
        self._current_batch = self._next_batch
        self._next_batch = next(self._iterator, None)

        assert self._previous_batch is not None

        # TODO: Return done=True when the iterator is exhausted?
        self._done = self._next_batch is None
        obs = self._current_batch[0]
        reward = self._previous_batch[1]
        # Empty for now I guess?
        info = {}
        return obs, reward, self._done, info

    def action(self, action: ActionType) -> ActionType:
        """ Transform the action, if needed.

        Parameters
        ----------
        action : ActionType
            [description]

        Returns
        -------
        ActionType
            [description]
        """
        return action
    
    def observation(self, observation: ObservationType) -> ObservationType:
        """ Transform the observation, if needed.

        Parameters
        ----------
        observation : ObservationType
            [description]

        Returns
        -------
        ObservationType
            [description]
        """
        return observation
    
    def reward(self, reward: RewardType) -> RewardType:
        """ Transform the reward, if needed.

        Parameters
        ----------
        reward : RewardType
            [description]

        Returns
        -------
        RewardType
            [description]
        """
        return reward
    
    def use_action(self, action):
        """ Override this method if you want the actions to affect the env
        somehow. (You could also just override the `step` method, I guess).
        
        Parameters
        ----------
        action : [type]
            [description]
        """

    def get_info(self) -> Dict:
        """Returns the dict to be returned as the 'info' in step().

        IDEA: We could subclass this to change whats in the 'info' dict, maybe
        add some task information?
        
        Returns
        -------
        Dict
            [description]
        """
        return {}

    def render(self, mode="rgb_array") -> np.ndarray:
        return self._current_batch.cpu().numpy()        
    
    def __iter__(self) -> Iterable[Tuple[ObservationType, Optional[RewardType]]]:
        """Iterate over the dataset, yielding batches of Observations and
        Rewards, just like a regular DataLoader.
        """
        if self.split_batch_fn:
            return map(self.split_batch_fn, super().__iter__())
        else:
            return super().__iter__()
        # for batch in super().__iter__():
        #     batch = self.batch_transforms(batch)
            
        #     # For now, just to simplify, we assume that the batch has already
        #     # been split into Observations and Actions by a SplitBatch transform.
        #     assert len(batch) == 2
        #     assert isinstance(batch[0], Observations)
        #     assert isinstance(batch[1], Rewards)
        #     self.observations, self.rewards = batch
            
        #     if self.pretend_to_be_active:
        #         # TODO: Should we yield one item, or two?
        #         yield self.observations, None
        #     else:
        #         yield self.observations, self.rewards

    def send(self, action: Actions) -> Rewards:
        """ Return the last latch of rewards from the dataset (which were
        withheld if in 'active' mode)
        """
        assert False, (
            "TODO: work in progress, if you're gonna pretend this is an "
            "'active' environment, then use the gym API for now."
        )
        # assert self.action_space.contains(action), action
        return None

    def close(self):
        pass
