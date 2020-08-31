""" Labeled, Unlabeled and Semi-supervised Replay buffer objects.

TODO: Unused for now, but could be used in a LightningModule.
"""
import random
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import InitVar, dataclass
from typing import *
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from simple_parsing.helpers.serialization.serializable import D
from common.loss import Loss
from simple_parsing import field, mutable_field
from utils.json_utils import Serializable, Pickleable
from utils.logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")

@dataclass
class ReplayBuffer(Deque[T], Pickleable):
    """Simple implementation of a replay buffer.

    Uses a doubly-ended Queue, which unfortunately isn't registered as a buffer
    for pytorch.
    """
    def __init__(self, capacity: int):
        super().__init__(maxlen=capacity)
        # self.extend("ABC")
        self.capacity: int = capacity
        # self.register_buffer("memory", torch.zeros(1)) # TODO: figure out how to set it with a Tensor maybe?
        self.labeled: Optional[bool] = None
        self.current_size: int = 0

    def as_dataset(self) -> TensorDataset:
        contents = zip(*self)
        return TensorDataset(*map(torch.stack, contents))

    def _push_and_sample(self, *values: T, size: int) -> List[T]:
        """Pushes `values` into the buffer and samples `size` samples from it.

        NOTE: In contrast to `push`, allows sampling more than `len(self)`
        samples from the buffer (up to `len(self) + len(values)`)

        Args:
            *values (T): An iterable of items to push.
            size (int): Number of samples to take.
        """
        extended = list(self)
        extended.extend(values)
        # NOTE: Type hints indicate that random.shuffle expects a list, not
        # a deque. Seems to work just fine though.
        random.shuffle(extended)  # type: ignore
        assert size <= len(extended), f"Asked to sample {size} values, while there are only {len(extended)} in the batch + buffer!"
        
        self.extend(extended)
        return extended[:size]

    def _sample(self, size: int) -> List[T]:
        assert size <= len(self), f"Asked to sample {size} values while there are only {len(self)} in the buffer!"
        return random.sample(self, size)

    @property
    def full(self) -> bool:
        return len(self) == self.capacity 

@dataclass
class UnlabeledReplayBuffer(ReplayBuffer[Tensor]):
    def sample_batch(self, size: int) -> Tensor:
        batch = super()._sample(size)
        return torch.stack(batch)

    def push(self, x_batch: Tensor, y_batch: Tensor=None) -> None:
        super()._push(x_batch)

    def push_and_sample(self, x_batch: Tensor, y_batch: Tensor=None, size: int=None) -> Tensor:
        size = x_batch.shape[0] if size is None else size
        return torch.stack(super()._push_and_sample(x_batch, size=size))


@dataclass
class LabeledReplayBuffer(ReplayBuffer[Tuple[Tensor, Tensor]]):
    def sample(self, size: int) -> Tuple[Tensor, Tensor]:
        list_of_pairs = super()._sample(size)
        data_list, target_list = zip(*list_of_pairs)
        return torch.stack(data_list), torch.stack(target_list)

    def push(self, x_batch: Tensor, y_batch: Tensor) -> None:
        super()._push(zip(x_batch, y_batch))

    def push_and_sample(self, x_batch: Tensor, y_batch: Tensor, size: int=None) -> Tuple[Tensor, Tensor]:
        size = x_batch.shape[0] if size is None else size
        list_of_pairs = super()._push_and_sample(*zip(x_batch, y_batch), size=size)
        data_list, target_list = zip(*list_of_pairs)
        return torch.stack(data_list), torch.stack(target_list)

    def samples_per_class(self) -> Dict[int, int]:
        """ Returns a Counter showing how many samples there are per class. """
        # TODO: Idea, could use the None key for unlabeled replay buffer.
        return Counter(int(y) for x, y in self)


class CoolReplayBuffer(object):
    def __init__(self, labeled_capacity: int, unlabeled_capacity: int=0):
        """Semi-Supervised (ish) version of a replay buffer.
        With the default parameters, acts just like a regular replay buffer.

        When passed `unlabeled_capacity`, allows for storing unlabeled samples
        as well as labeled samples. Unlabeled samples are stored in a different
        buffer than labeled samples.

        Allows sampling both labeled and unlabeled samples.

        Args:
            labeled_capacity (int): [description]
            unlabeled_capacity (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self.labeled_capacity = labeled_capacity
        self.unlabeled_capacity = unlabeled_capacity

        self.labeled = LabeledReplayBuffer(labeled_capacity)
        self.unlabeled = UnlabeledReplayBuffer(unlabeled_capacity)

    def sample(self, size: int) -> Tuple[Tensor, Tensor]:
        """Takes `size` (labeled) samples from the buffer.

        Args:
            size (int): Number of samples to return.

        Returns:
            Tuple[Tensor, Tensor]: batched data and label tensors.
        """
        assert size <= len(self.labeled), (
            f"Asked to sample {size} values while there are only "
            f"{len(self.labeled)} labeled samples in the buffer! "
        )
        return self.labeled.sample(size)

    def sample_unlabeled(self, size: int, take_from_labeled_buffer_first: bool=None) -> Tensor:
        """Samples `size` unlabeled samples.

        Can also use samples from the labeled replay buffer (while discarding
        the labels) if there is no unlabeled replay buffer.

        Args:
            size (int): Number of x's to sample
            take_from_labeled_buffer_first (bool, optional):
                When `None` (default), doesn't take any samples from the labeled
                buffer.
                When `True`, prioritizes taking samples from the labeled replay
                buffer.
                When `False`, prioritizes taking samples from the unlabeled replay
                buffer, but take the remaining samples from the labeled buffer.

        Returns:
            Tensor: A batch of X's.
        """
        
        total = len(self.unlabeled)
        if take_from_labeled_buffer_first is not None:
            total += len(self.labeled)

        assert size <= total, (
            f"Asked to sample {size} values while there are only "
            f"{total} unlabeled samples in total in the buffer! "
        )
        # Number of x's we still have to sample.
        samples_left = size
        tensors: List[Tensor] = []

        if take_from_labeled_buffer_first:
            # Take labeled samples and drop the label.
            n_samples_from_labeled = min(len(self.labeled), samples_left)
            if n_samples_from_labeled > 0:
                data, _ = self.labeled.sample(size)
                samples_left -= data.shape[0]
                tensors.append(data)
        
        # Take the rest of the samples from the unlabeled buffer.
        n_samples_from_labeled = min(len(self.labeled), samples_left)
        data = self.unlabeled.sample_batch(samples_left) 
        tensors.append(data)
        samples_left -= data.shape[0]

        if take_from_labeled_buffer_first is False:
            # Take the rest of the labeled samples and drop the label.
            n_samples_from_labeled = min(len(self.labeled), samples_left)
            if n_samples_from_labeled > 0:
                data, _ = self.labeled.sample(size)
                samples_left -= data.shape[0]
                tensors.append(data)

        data = torch.cat(tensors)
        return data

    def push_and_sample(self, x: Tensor, y: Tensor, size: int=None) -> Tuple[Tensor, Tensor]:
        size = x.shape[0] if size is None else size
        self.unlabeled.push(x)
        return self.labeled.push_and_sample(x, y, size=size)
        
    def push_and_sample_unlabeled(self, x: Tensor, y: Tensor=None, size: int=None) -> Tensor:
        size = x.shape[0] if size is None else size
        if y is not None:
            self.labeled.push(x, y)
        return self.unlabeled.push_and_sample(x, size=size)
    
    def clear(self):
        self.labeled.clear()
        self.unlabeled.clear()


@dataclass
class ReplayOptions(Serializable):
    """ Options related to Replay. """
    # Size of the labeled replay buffer.
    labeled_buffer_size: int = field(0, alias="replay_buffer_size")
    # Size of the unlabeled replay buffer.
    unlabeled_buffer_size: int = 0

    # Always use the replay buffer to help "smooth" out the data stream.
    always_use_replay: bool = False
    # Sampling size, when used as described above to smooth out the data stream.
    # If not given, will use the same value as the batch size.
    sampled_batch_size: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.labeled_buffer_size > 0 or self.unlabeled_buffer_size > 0

