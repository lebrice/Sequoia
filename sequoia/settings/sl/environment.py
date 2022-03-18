"""TODO: Creates a Gym Environment (and DataLoader) from a traditional
Supervised dataset. 
"""

from collections import deque
from typing import *

import gym
import numpy as np
from gym import spaces
from gym.vector.utils import batch_space
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter

from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.gym_wrappers.utils import tile_images
from sequoia.common.spaces import Image
from sequoia.common.transforms import Transforms
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PassiveEnvironment(
    DataLoader,
    Environment[Tuple[ObservationType, Optional[ActionType]], ActionType, RewardType],
):
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

    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        dataset: Union[IterableDataset, Dataset],
        split_batch_fn: Callable[[Tuple[Any, ...]], Tuple[ObservationType, ActionType]] = None,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        reward_space: gym.Space = None,
        n_classes: int = None,
        pretend_to_be_active: bool = False,
        strict: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """Creates the DataLoader/Environment for the given dataset.

        Parameters
        ----------
        dataset : Union[IterableDataset, Dataset]
            The dataset to iterate on. Should ideally be indexable (a Map-style
            dataset).

        split_batch_fn : Callable[ [Tuple[Any, ...]], Tuple[ObservationType, ActionType] ], optional
            A function to call on each item in the dataset in order to split it into
            Observations and Rewards, by default None, in which case we assume that the
            dataset items are tuples of length 2.

        observation_space : gym.Space, optional
            The single (non-batched) observation space. Default to `None`, in which case
            this will try to infer the shape of the space using the first item in the
            dataset.

        action_space : gym.Space, optional
            The non-batched action space. Defaults to None, in which case the
            `n_classes` argument must be passed, and the action space is assumed to be
            discrete (i.e. that the loader is for a classification dataset).

        reward_space : gym.Space, optional
            The non-batched reward (label) space. Defaults to `None`, in which case it
            will be the same as the action space (as is the case in classification).

        n_classes : int, optional
            Number of classes in the dataset. Used in case `action_space` isn't passed.
            Defaults to `None`.

        pretend_to_be_active : bool, optional
            Wether to withhold the rewards (labels) from the batches when being
            iterated on like the usual dataloader, and to only give them back
            after an action is received through the 'send' method. False by
            default, in which case this behaves exactly as a normal dataloader
            when being iterated on.

            When False, the batches yielded by this dataloader will be of the form
            `Tuple[Observations, Rewards]` (as usual in SL).
            However, when set to True, the batches will be `Tuple[Observations, None]`!
            Rewards will then be returned by the environment when an action is passed to
            the Send method.

        strict : bool, optional
            [description], by default False

        # Examples:
        ```python
        train_env = PassiveEnvironment(MNIST("data"), batch_size=32, num_classes=10)

        # The usual Dataloader-style:
        for x, y in train_env:
            # train as usual
            (...)

        # OpenAI Gym style:
        for episode in range(5):
            # NOTE: "episode" in RL is an "epoch" in SL:
            obs = train_env.reset()
            done = False
            while not done:
                actions = train_env.action_space.sample()
                obs, rewards, done, info = train_env.step(actions)
        ```
        """
        super().__init__(dataset=dataset, drop_last=drop_last, **kwargs)
        self.split_batch_fn = split_batch_fn

        # TODO: When the spaces aren't passed explicitly, assumes a classification dataset.
        if not observation_space:
            # NOTE: Assuming min/max of 0 and 1 respectively, but could actually use
            # min_max of the dataset samples too.
            first_item = self.dataset[0]
            if isinstance(first_item, tuple):
                x, *_ = first_item
            else:
                assert isinstance(first_item, (np.ndarray, Tensor))
                x = first_item
            observation_space = Image(0.0, 1.0, x.shape)
        if not action_space:
            assert n_classes, "must pass either `action_space`, or `n_classes` for now"
            action_space = spaces.Discrete(n_classes)
        elif isinstance(action_space, spaces.Discrete):
            n_classes = action_space.n

        if not reward_space:
            # Assuming a classification dataset by default:
            # (action space = reward space = Discrete(n_classes))
            reward_space = action_space

        assert observation_space
        assert action_space
        assert reward_space

        self.single_observation_space: gym.Space = observation_space
        self.single_action_space: gym.Space = action_space
        self.single_reward_space: gym.Space = reward_space

        if self.batch_size:
            observation_space = batch_space(observation_space, self.batch_size)
            action_space = batch_space(action_space, self.batch_size)
            reward_space = batch_space(reward_space, self.batch_size)

        self.observation_space: gym.Space = add_tensor_support(observation_space)
        self.action_space: gym.Space = add_tensor_support(action_space)
        self.reward_space: gym.Space = add_tensor_support(reward_space)

        self.pretend_to_be_active = pretend_to_be_active
        self._strict = strict
        self._reward_queue = deque(maxlen=10)

        self.n_classes: Optional[int] = n_classes
        self._iterator: Optional[_BaseDataLoaderIter] = None
        # NOTE: These here are never processed with self.observation or self.reward.
        self._previous_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._current_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._next_batch: Optional[Tuple[ObservationType, RewardType]] = None
        self._done: Optional[bool] = None
        self._is_closed: bool = False

        self._action: Optional[ActionType] = None
        # from gym.envs.classic_control.rendering import SimpleImageViewer
        self.viewer = None

    def is_closed(self) -> bool:
        return self._is_closed

    def reset(self) -> ObservationType:
        """Resets the env by deleting and re-creating the dataloader iterator.

        TODO: This might be pretty expensive, since it's maybe re-creating all the
        worker processes. There might be an easier way of going about this.

        Returns the first batch of observations.
        """
        if self._is_closed:
            raise gym.error.ClosedEnvironmentError("Can't reset: Env is closed.")
        self._iterator = super().__iter__()
        self._previous_batch = None
        self._current_batch = self.get_next_batch()
        self._done = False
        obs = self._current_batch[0]
        return self.observation(obs)

    def close(self) -> None:
        if not self._is_closed:
            if self.viewer:
                self.viewer.close()
            if self.num_workers > 0 and self._iterator:
                self._iterator._shutdown_workers()
            self._is_closed = True

    def __del__(self):
        if not self._is_closed:
            self.close()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        observations = self._current_batch[0]
        if isinstance(observations, Observations):
            image_batch = observations.x
        else:
            assert isinstance(observations, Tensor)
            image_batch = observations
        if isinstance(image_batch, Tensor):
            image_batch = image_batch.cpu().numpy()

        if self.batch_size:
            image_batch = tile_images(image_batch)

        image_batch = Transforms.channels_last_if_needed(image_batch)
        image_batch = Transforms.three_channels(image_batch)
        assert image_batch.shape[-1] in {3, 4}, image_batch.shape
        if image_batch.dtype == np.float32:
            assert (0 <= image_batch).all() and (image_batch <= 1).all()
            image_batch = (256 * image_batch).astype(np.uint8)
        assert image_batch.dtype == np.uint8

        if mode == "rgb_array":
            # NOTE: Need to create a single image, channels_last format, and
            # possibly even of dtype uint8, in order for things like Monitor to
            # work.
            return image_batch

        if mode == "human":
            # return plt.imshow(image_batch)
            if self.viewer is None:
                display = None
                # TODO: There seems to be a bit of a bug, tests sometime fail because
                # "Can't connect to display: None" etc.
                from gym.utils import pyglet_rendering
                # from pyvirtualdisplay import Display
                # display = Display(visible=0, size=(1366, 768))
                # display.start()
                self.viewer = pyglet_rendering.SimpleImageViewer()

            self.viewer.imshow(image_batch)
            return self.viewer.isopen

        raise NotImplementedError(f"Unsuported mode {mode}")

    def get_next_batch(self) -> Tuple[ObservationType, RewardType]:
        """Gets the next batch from the underlying dataset.

        Uses the `split_batch_fn`, if needed. Does NOT apply the self.observation
        and self.reward methods.

        Returns
        -------
        Tuple[ObservationType, RewardType]
            [description]
        """
        if self._is_closed:
            raise gym.error.ClosedEnvironmentError("Can't get the next batch: Env is closed.")
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration:
            batch = None

        if self.split_batch_fn and batch is not None:
            batch = self.split_batch_fn(batch)
        return batch
        # obs, reward = batch
        # return self.observation(obs), self.reward(reward)

    def step(self, action: ActionType) -> Tuple[ObservationType, RewardType, bool, Dict]:
        if self._is_closed:
            raise gym.error.ClosedEnvironmentError("Can't step on a closed env.")
        if self._done is None:
            raise gym.error.ResetNeeded("Need to reset the env before calling step.")
        if self._done:
            raise gym.error.ResetNeeded("Need to reset the env since it is done.")

        # Transform the Action, if needed:
        action = self.action(action)

        # NOTE: This prev/current/next setup is so we can give the right 'done'
        # signal.
        self._previous_batch = self._current_batch
        if self._next_batch is None:
            # This should only ever happen right after resetting.
            self._next_batch = self.get_next_batch()
        self._current_batch = self._next_batch
        self._next_batch = self.get_next_batch()
        # self._next_batch = self._observations, self._rewards

        assert self._previous_batch is not None

        # TODO: Return done=True when the iterator is exhausted?
        self._done = self._next_batch is None
        obs = self._current_batch[0]
        reward = self._previous_batch[1]
        # Empty for now I guess?
        info = {}
        return obs, reward, self._done, info

    def action(self, action: ActionType) -> ActionType:
        """Transform the action, if needed.

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
        """Transform the observation, if needed.

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
        """Transform the reward, if needed.

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

    def __iter__(self) -> Iterable[Tuple[ObservationType, Optional[RewardType]]]:
        """Iterate over the dataset, yielding batches of Observations and
        Rewards, just like a regular DataLoader.
        """
        # if self.split_batch_fn:
        #     return map(self.split_batch_fn, super().__iter__())
        # else:
        #     return super().__iter__()
        if self._is_closed:
            raise gym.error.ClosedEnvironmentError("Can't iterate over closed env.")

        for batch in super().__iter__():

            if self.split_batch_fn:
                observations, rewards = self.split_batch_fn(batch)
            else:
                if len(batch) != 2:
                    raise RuntimeError(
                        f"You need to pass a `split_batch_fn` to create "
                        f"observations and rewards, since batch doesn't have "
                        f"2 items: {batch}"
                    )
                observations, rewards = batch

            # Apply any transformations (in case this is wrapped with
            # TransformObservation or something similar)
            self._observations = self.observation(observations)
            self._rewards = self.reward(rewards)

            self._previous_batch = self._current_batch
            self._current_batch = (self._observations, self._rewards)

            if self.pretend_to_be_active:
                self._action = None
                self._reward_queue.append(self._rewards)
                yield self._observations, None
                if self._action is None:
                    if self._strict:
                        # IDEA: yield the same observation, as long as we dont receive an action.
                        raise RuntimeError("Need to send an action between each observations.")
                    logger.warning("Didn't receive an action, rewards will be delayed!.")
            else:
                yield self._observations, self._rewards

    def send(self, action: Actions) -> Rewards:
        """Return the last latch of rewards from the dataset (which were
        withheld if in 'active' mode)
        """
        if self.pretend_to_be_active:
            self._action = action
            return self._reward_queue.popleft()
        else:
            # NOTE: What about sending the reward as well this way?
            return self._rewards
