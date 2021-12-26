from collections import deque
from gym.vector.utils import shared_memory
from gym.vector.utils.shared_memory import create_shared_memory
import numpy as np
from torch.utils.data import DataLoader
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
    List,
    Union,
    overload,
)

from .episode import Episode, T, Transition

T = TypeVar("T")

from sequoia.methods.experience_replay import Buffer

from collections.abc import Iterable as _Iterable
from sequoia.common.typed_gym import _Space
from sequoia.utils.generic_functions import get_slice, set_slice, stack, concatenate

# NOTE: Usign this, but it would probably be easier to use arrays instead, no need for this to be
# shared memory at all.
# TODO: Register variants of these functions for writing/reading tensors rather than numpy arrays.
from gym.vector.utils import (
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
)
from torch.utils.data import IterableDataset

from gym.vector.utils.spaces import batch_space
import random

Item = TypeVar("Item", covariant=True)


class ReplayBuffer(IterableDataset[Item], Sequence[Item]):
    def __init__(self, item_space: _Space[Item], capacity: int, seed: int = None):
        super().__init__()
        self.item_space = item_space
        self._capacity = capacity
        # TODO: Make this equal to `register_buffer` somehow when the space is a TensorSpace or something.
        # self._data = create_shared_memory(self.item_space, n=self.capacity)
        # TODO: Could also maybe do this to create the buffers, allowing for batch read/write!
        self._data = create_empty_array(item_space, n=capacity)

        self._current_index = 0
        # Number of total insertions so far (can be greater than capacity).
        self._n_insertions = 0
        self.rng: np.random.RandomState
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.RandomState(seed=seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def full(self) -> bool:
        return len(self) == self.capacity

    def __len__(self) -> int:
        if self._n_insertions < self.capacity:
            # Not full yet.
            return self._n_insertions
        # Full:
        return self.capacity

    def __setitem__(self, index: int, value: Item) -> None:
        # write_to_shared_memory(self.item_space, index=index, value=value, shared_memory=self._data)
        # return
        if isinstance(index, int):
            if not self.full and index == self._current_index:
                # Let it slide:
                # TODO: There are some bugs here because set_slice expects indices to be arrays of ints.
                set_slice(self._data, indices=index, values=value)
            elif not (0 <= index < len(self)):
                raise IndexError(index)
            else:
                # batched_value = stack(value)
                set_slice(self._data, indices=index, values=value)
        else:
            # write_to_shared_memory(self.item_space, index=index, value=value, shared_memory=self._data)
            set_slice(self._data, index=index, values=value)

    def __getitem__(self, index: int) -> Item:
        if isinstance(index, int) and not (0 <= index < len(self)):
            # TODO: Allow negative indices
            raise IndexError(index)
        # return read_from_shared_memory(self.item_space, index=index, shared_memory=self._data)
        if isinstance(index, int):
            # NOTE: This kinda makes sense: Get a "batched" item, using a batched version of the
            # index, then take a slice of the result:
            batch = get_slice(self._data, indices=[index])
            return get_slice(batch, indices=[0])
        else:
            return get_slice(self._data, indices=index)

    def append(self, item: Item) -> None:
        # Behaves like a deque when using append/extend by default.
        self._n_insertions += 1
        self._current_index += 1
        self._current_index %= self.capacity
        self[self._current_index] = item

    def sample(self, n_samples: int = None) -> Union[Item, Sequence[Item]]:
        if n_samples is None:
            return self[self.rng.choice(len(self), 1)]
        indices = self.rng.choice(len(self), n_samples, replace=False)
        # TODO: Would be better to do batch read/write, for sure.
        return self[indices]
        # return [self[i] for i in indices]

    def extend(self, items: Iterable[Item]) -> None:
        # NOTE: Should this just redirect to add_reservoir?
        # for item in items:
        #     self.append(item)
        self.add_reservoir(items)

    def add_reservoir(self, batch: Iterable[Item]) -> None:
        batch_length = len(batch)

        items_to_add = list(batch)
        n = len(batch)
        # Adapted from https://en.wikipedia.org/wiki/Reservoir_sampling#Simple_algorithm :
        # for i, item in enumerate(batch):
        #     if self.full:
        #         break
        #     else:
        #         self.append(item)
        # # Handle the rest:
        # # NOTE: Start form the last value of i (from the previous loop.)
        # for i, item in range(i, n):
        #     write_index = random.randrange(i)
        #     if write_index < self.capacity:
        #         self[write_index] = item

        # OR Taken from https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm :

        for i, item in enumerate(batch):
            if self.full:
                break
            self.append(item)
        # NOTE: i is still usable here.

        # random() generates a uniform (0,1)
        W = np.exp(np.log(self.rng.random()) / self.capacity)
        while i <= n:
            # i := i + floor(log(random())/log(1-W)) + 1
            # TODO: Increment I by at least 1? What is the other term?
            i += 1 + int(np.floor(np.log(self.rng.random()) / np.log(1 - W)))
            if i < n:
                # (* replace a random item of the reservoir with item i *)
                # R[randomInteger(1,k)] := S[i]  // random index between 1 and k, inclusive
                # replace a random item of the reservoir with item i
                # random index between 0 and k-1, inclusive
                write_index = self.rng.randint(0, self.capacity)
                self[write_index] = batch[i]

                # W := W * exp(log(random())/k)
                W *= np.exp(np.log(self.rng.random()) / self.capacity)
        return

    @overload
    def __add__(
        self: "ReplayBuffer[Item]", other: "ReplayBuffer[Item]"
    ) -> "ReplayBuffer[Item]":
        ...

    @overload
    def __add__(
        self: "ReplayBuffer[Item]", other: "ReplayBuffer[T]"
    ) -> "ReplayBuffer[Union[Item, T]]":
        ...

    def __add__(
        self: "ReplayBuffer[Item]",
        other: Union["ReplayBuffer[Item]", "ReplayBuffer[T]", Any],
    ) -> Union["ReplayBuffer[Union[Item, T]]", "ReplayBuffer[Item]"]:
        raise NotImplementedError("IDEA: add two buffers?")
