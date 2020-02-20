""" Set of Utilities. """


import torch
from torch import nn, Tensor

from typing import List, Deque, Optional
from collections import deque
import collections

from collections.abc import MutableMapping
from typing import MutableMapping

class TensorCache(MutableMapping[Tensor, Tensor]):
    """A mutable mapping of individual (not batched) tensors to their outputs.
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.x_shape: Optional[torch.Size] = None
        self.y_shape: Optional[torch.Size] = None
        self.x_cache: Optional[Tensor] = None
        self.y_cache: Optional[Tensor] = None
        self.head: int = 0
        self.tail: int = 0

    def __getitem__(self, key: Tensor) -> Tensor:
        if self.y_cache is None:
            self.x_shape = key.shape
            raise KeyError(key)
   
    def __setitem__(self, key: Tensor, value: Tensor) -> None:
        if self.x_cache is None:
            self.x_shape = key.shape
            self.x_cache = key.new_zeros([self.capacity, *self.x_shape])
        if self.y_cache is None:
            self.y_shape = value.shape
            self.y_cache = value.new_zeros([self.capacity, *self.y_shape])
        self.head += 1
        self.head %= self.capacity
        self.x_cache[self.head] = key
        self.y_cache[self.head] = value
    
    def __len__(self) -> int:
        return ((self.head + self.capacity) - self.tail) % self.capacity

    def __delitem__(self, key: Tensor) -> None:
        pass

    def __iter__(self):
        return iter(self.x_cache)



class CachedForwardPass(nn.Module):
    """TODO: create a wrapper that can cache results of a forward pass
    Thoughts:
    - Makes no sense, since whenever the model updates you will clear the cache anyway!
    - Use a tensor to hold the results? or use deque? 
    - Check membership at an example level?
    - How should this be used? as a decorator on the forward method? or as a base class?
    """
    def __init__(self, capacity: int = 100):
        super().__init__()
        self.capacity = capacity
        self.cached_x: Deque[Tensor] = deque(maxlen=capacity)
        self.cached_y: Deque[Tensor] = deque(maxlen=capacity)
        self.head: int = 0
        self.tail: int = 0

    def forward(self, x_batch: Tensor):
        is_cached = [x in self for x in x_batch]
        print(is_cached)

    def __contains__(self, item: Tensor) -> bool:
        pass
    
    def __getitem__(self, key: Tensor) -> Tensor:
        pass

    def __setitem__(self, key: Tensor, value: Tensor) -> None:
        pass

    def create_cache(self, example_x: Tensor, example_y: Tensor) -> None:
        self.cache_x = example_x.new_zeros([self.capacity, *example_x.shape])
        self.cache_y = example_y.new_zeros([self.capacity, *example_y.shape])

    def clear(self) -> None:
        self.cached_x.clear()
        self.cached_y.clear()
            
cache = TensorCache(5)



d = TensorCache(5)
zero = torch.zeros(3,3)
one = torch.ones(3,3)
batch = torch.stack([zero, one])
print([t in d for t in batch.unbind()])
d[zero] = torch.Tensor(123)
print(torch.zeros(3,3) in d)
print(one in d)
