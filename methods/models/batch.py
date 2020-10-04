""" WIP (@lebrice): Playing around with the idea of using a typed object to represent
the different forms of "batches" that different settings produce and that
different models expect.
"""
import dataclasses
import itertools
from collections import abc as collections_abc
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import torch
from torch import Tensor

# WIP (@lebrice): Playing around with this idea, to try and maybe use of typed
# objects for the 'Observation', the 'Action' and the 'Reward' for each kind of
# model. Might be a bit too complicated though.

Item = TypeVar("Item")

@dataclass(frozen=True)
class Batch(Sequence[Item]):
    field_names: ClassVar[Tuple[str, ...]]

    def __post_init__(self):
        type(self).field_names = [f.name for f in dataclasses.fields(self)]
        
    def __iter__(self) -> Iterable[Item]:
        for name in self.field_names:
            yield getattr(self, name)
        return iter(self.as_tuple())

    def __len__(self):
        return len(self.field_names)

    def __getitem__(self, index: Union[int, str, slice]):
        if isinstance(index, int):
            field_name = self.field_names[index]
            return getattr(self, field_name)
        elif isinstance(index, slice):
            field_names = self.field_names[index]
            return tuple(self[field_name] for field_name in field_names)
        elif isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, collections_abc.Iterable):
            return tuple(self[item] for item in index)
        raise IndexError(index)

    def __setitem__(self, index: Union[int, str, slice], value: Any):
        # NOTE: If we mark this dataclass as frozen, then this won't work.
        if isinstance(index, int):
            field_name = self.field_names[index]
            return setattr(self, field_name, value)
        elif isinstance(index, slice):
            field_names = self.field_names[index]
            if not isinstance(value, collections_abc.Sized):
                # set the same value at all indices.
                value = itertools.cycle([value])
            elif len(value) != len(field_names):
                raise RuntimeError(
                    f"Can't set value of {value} at index {index}: should have "
                    f"received {len(self)} indices or a value that isn't sized."
                )
            assert isinstance(value, collections_abc.Sized)
            
            for field_name, value in zip(self.field_names[index], value):
                set
            return tuple(self[field_name] for field_name in field_names)
        elif isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, collections_abc.Iterable):
            return tuple(self[item] for item in index)
    
    def items(self) -> Iterable[Tuple[str, Item]]:
        for name in self.field_names:
            yield name, getattr(self, name)
    
    @property
    def device(self) -> Optional[torch.device]:
        """ Returns the device common to all the elements in the Batch, else
        None if there is no consensus. """
        device = None
        for tensor in self:
            if not hasattr(tensor, "device"):
                continue
            if device is None:
                device = tensor.device
            if device != tensor.device:
                return None # No consensus on the device
        return device

    def as_tuple(self) -> Tuple[Item, ...]:
        return dataclasses.astuple(self)

    def as_dict(self) -> Dict[str, Item]:
        return dataclasses.asdict(self)

    def to(self, *args, **kwargs):
        return type(self)(*(
            item.to(*args, **kwargs) if isinstance(item, Tensor) else item
            for item in self
        ))
    
    def shapes(self) -> Tuple[Optional[torch.Size], ...]:
        """ Returns a tuple of the shapes of the elements in the batch. 
        If an element doesn't have a 'shape' attribute, returns None for that
        element.
        """
        return tuple(getattr(item, "shape", None) for item in self)
    
    @property
    def batch_size(self) -> int:
        return self[0].shape[0]

    @classmethod
    def from_inputs(cls, inputs):
        """ Converts a batch of items into a 'Batch' object. """
        if isinstance(inputs, cls):
            return inputs
        if isinstance(inputs, Tensor):
            return cls(inputs)
        if isinstance(inputs, dict):
            return cls(**inputs)
        if isinstance(inputs, (tuple, list)):
            return cls(*inputs)
        raise RuntimeError(
            f"Don't know how to turn inputs {inputs} (type {type(inputs)}) "
            f"into a Batch object of type {cls}!"
        )
        return cls(inputs)
    

# TODO: FIXME: We should probably be using something like an Observation / 
# action / reward space! Would that replace or complement these objects?
# Maybe we could actually add a `space` @property on these? Is there such a
# thing as 'optional' dimensions in gym Spaces?


@dataclass(frozen=True)
class Observation(Batch):
    x: Tensor


@dataclass(frozen=True)
class Action(Batch):
    # Predictions from the model in a supervised setting, or chosen action
    # in an RL setting.
    y_pred: Tensor


@dataclass(frozen=True)
class Reward(Batch):
    y: Optional[Tensor]

