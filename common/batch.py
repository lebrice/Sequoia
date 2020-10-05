""" WIP (@lebrice): Playing around with the idea of using a typed object to
represent the different forms of "batches" that settings produce and that
different models expect.
"""
from abc import ABC
import dataclasses
import itertools
from collections import abc as collections_abc
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union, Mapping, Iterator, Set)

import torch
from torch import Tensor

Item = TypeVar("Item", bound=collections_abc.Sized)


@dataclass(frozen=True)
class Batch(Sequence[Item], ABC):
    """A typed version of a batch of tensors.

    Essentially a qwerky mix of NamedTuple, Dict, and dataclass, with some
    Tensor-like methods like `move()`.
    """
    field_names: ClassVar[Tuple[str, ...]]

    def __post_init__(self):
        type(self).field_names = [f.name for f in dataclasses.fields(self)]
    
    def __init_subclass__(cls, *args, **kwargs):
        # IDEA: By not marking 'Batch' a dataclass, we would let the subclass
        # decide it if wants to be frozen or not!
        if not dataclasses.is_dataclass(cls):
            raise RuntimeError(f"{__class__} subclass {cls} must be a dataclass!")
        # Subclasses of `Batch` should be dataclasses!
        super().__init_subclass__(*args, **kwargs)

    def __iter__(self) -> Iterator[Item]:
        for name in self.field_names:
            yield getattr(self, name)
        yield from iter(self.as_tuple())

    def __len__(self):
        return len(self.field_names)

    def __getitem__(self, index: Union[int, str]):  # type: ignore
        if isinstance(index, int):
            field_name = self.field_names[index]
            return getattr(self, field_name)
        elif isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, slice):
            # I don't think it would be a good idea to support slice indexing,
            # as it could be confusing and give the user the impression that it
            # is slicing into the tensors, rather than into the fields.
            # Plus, there really shouldn't be that many fields in a Batch object
            # anyway.
            raise NotImplementedError(f"Batch doesn't support slice indexing.")
        raise IndexError(index)

    def __setitem__(self, index: Union[int, str], value: Any):
        # NOTE: If we mark this dataclass as frozen, then this won't work.
        if isinstance(index, int):
            field_name = self.field_names[index]
            return setattr(self, field_name, value)
        elif isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, slice):
            # I don't think it would be a good idea to support slice indexing,
            # as it could be confusing and give the user the impression that it
            # is slicing into the tensors, rather than into the fields. Plus,
            # There really shouldn't be that many fields in a Batch object.
            raise NotImplementedError(f"Batch doesn't support slice indexing.")
        raise IndexError(index)

    def keys(self) -> Set[str]:
        return set(self.field_names)
    
    def values(self) -> Iterable[Item]:
        for name in self.field_names:
            yield getattr(self, name)

    def items(self) -> Iterable[Tuple[str, Item]]:
        for name in self.field_names:
            yield name, getattr(self, name)
    
    @property
    def device(self) -> Optional[torch.device]:
        """ Returns the device common to all the elements in the Batch, else
        None if there is no consensus. """
        device = None
        for item in self.values():
            item_device = getattr(item, "device", None)
            if item_device is not None and device is None:
                device = item_device
            if item_device != device:
                # No consensus on the devices, so return None.
                return None
        return device

    def as_tuple(self) -> Tuple[Item, ...]:
        return dataclasses.astuple(self)

    def as_dict(self) -> Dict[str, Item]:
        return dataclasses.asdict(self)

    def to(self, *args, **kwargs):
        return type(self)(**{
            name: item.to(*args, **kwargs) if isinstance(item, Tensor) else item
            for name, item in self.items()
        })
    
    @property
    def shapes(self) -> Tuple[Optional[Union[torch.Size, Tuple[int, ...]]], ...]:
        """ Returns a tuple of the shapes of the elements in the batch. 
        If an element doesn't have a 'shape' attribute, returns None for that
        element.
        """
        return tuple(getattr(item, "shape", None) for item in self.values())
    
    @property
    def batch_size(self) -> Optional[int]:
        """ Returns the batch size, i.e. the length of the first dimension of
        all non-None items in the batch. If there are no non-None items in the
        batch, returns None.
        """
        for item in self.values():
            if item is not None:
                return len(item)
        return None
    
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
