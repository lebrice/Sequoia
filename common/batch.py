""" WIP (@lebrice): Playing around with the idea of using a typed object to
represent the different forms of "batches" that settings produce and that
different models expect.
"""
import dataclasses
import itertools
from abc import ABC
from collections import abc as collections_abc
from collections import namedtuple
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, Generic, Iterable, Iterator, List,
                    Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Type,
                    TypeVar, Union)

import gym
import numpy as np
import torch
from gym import spaces
from torch import Tensor

from utils.slicing import get_slice, set_slice
Item = TypeVar("Item", bound=collections_abc.Sized)


@dataclass(frozen=True)
class Batch(ABC):
    """ABC for objects that represent a batch of tensors. Behaves like tuple.

    Can be used the same way as a tuple of tensors, or an immutable mapping from
    strings to tensors.
    Also has some Tensor-like methods like to(), numpy(), detach(), etc.
    Supports tuple and indexing also.
    
    NOTE: One difference with dict is that __iter__ gives an
    iterator over the values, not the string keys.
    
    Examples:

    >>> import torch
    >>> from typing import Optional
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class MyBatch(Batch):
    ...     x: Tensor
    ...     y: Tensor = None
    >>> batch = MyBatch(x=torch.ones([10, 3, 32, 32]), y=torch.arange(10))
    >>> batch.shapes
    (torch.Size([10, 3, 32, 32]), torch.Size([10]))
    >>> batch.batch_size
    10
    >>> batch.dtypes
    (torch.float32, torch.int64)
    >>> batch.dtype # No shared dtype, so dtype returns None.
    >>> batch.float().dtype # Converting the all items to float dtype:
    torch.float64
    
    Device-related methods:
    
    >>> batch.device  # Returns the device common to all items, or None.
    device(type='cpu')
    >>> batch.to("cuda").device
    device(type='cuda', index=0)
    """
    # TODO: Would it make sense to add a gym Space classvar here? 
    # space: ClassVar[Optional[gym.Space]] = None
    field_names: ClassVar[List[str]]
    _namedtuple: ClassVar[Type[NamedTuple]]
    
    def __init_subclass__(cls, *args, **kwargs):
        # IDEA: By not marking 'Batch' a dataclass, we would let the subclass
        # decide it if wants to be frozen or not!
        
        # Subclasses of `Batch` should be dataclasses!
        if not dataclasses.is_dataclass(cls):
            raise RuntimeError(f"{__class__} subclass {cls} must be a dataclass!")
        super().__init_subclass__(*args, **kwargs)

    def __post_init__(self):
        # TODO: We have to set these here because __init_subclass__ is called
        # before the dataclasses package sets the 'fields' attribute, it seems.
        # Create a NamedTuple type for this new subclass.
        cls = type(self)
        if "field_names" not in cls.__dict__:
            type(self).field_names = [f.name for f in dataclasses.fields(self)]
        if "_named_tuple" not in cls.__dict__:
            type(self)._namedtuple = namedtuple(type(self).__name__ + "Tuple", self.field_names)

    def unwrap(self) -> Union[Item, Tuple[Item, ...]]:
        """ Returns the 'unwrapped' contents of this object, which will be a
        tuple of batched tensors if there is more than one field, or a single
        batched tensor is there is only one field.
        """
        tensors = self.as_namedtuple()
        return tensors[0] if len(tensors) == 1 else tensors 

    # def __iter__(self) -> Iterable[Tuple[Item, ...]]:
    #     # return itertools.starmap(self._namedtuple, zip(*self.as_tuple()))
    #     return iter(self.as_tuple())
    
        # for name in self.field_names:
        #     yield getattr(self, name)

    # def __len__(self):
    #     return len(self.field_names)

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
            if index == slice(None, None, None):
                return self
            raise NotImplementedError(
                "Batch objects only support slice indexing with empty slices."
            )
        elif index is Ellipsis:
            return self
        elif isinstance(index, (tuple, list)):
            field_index = index[0]
            if len(index) <= 1:
                raise IndexError(f"Invalid index {index}: When indexing with "
                                 f"tuples, they need to have len > 1.")
            if isinstance(field_index, int):
                return self[field_index][index[1:]]

            # e.g: forward_pass[:, 1]
            if field_index == slice(None):
                # Get all fields, sliced with the second item.
                return get_slice(self, index)
                # return tuple(value[index[1:]] if value is not None else None 
                #             for value in self.values())
            
            raise NotImplementedError(f"{type(self)} doesn't support indexing with {index}.")
            return self[field_index][index[1:]]
            # else:
            #     raise IndexError("Can only use int or empty slice as first "
            #                      "item of index tuple.")
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
    def devices(self) -> Tuple[Optional[torch.device]]:
        """ Returns the device common to all the elements in the Batch, else
        None if not all items share the same device. """
        return tuple(
            getattr(value, "device", None) for value in self.values()
        )

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the device common to all items, or None.

        Returns
        -------
        Tuple[Optional[torch.device]]
            None if the devices are unknown/different, or the common device.
        """
        devices = self.devices
        if not devices or not all(device == devices[0] for device in devices):
            # No common dtype.
            return None
        return devices[0]

    @property
    def dtypes(self) -> Tuple[Optional[torch.dtype]]:
        return tuple(getattr(value, "dtype", None) for value in self.values())

    @property
    def dtype(self) -> Tuple[Optional[torch.dtype]]:
        """Returns the dtype common to all tensors, or None.

        Returns
        -------
        Tuple[Optional[torch.dtype]]
            None if the dtypes are unknown/different, or the common dtype.

        Raises
        ------
        NotImplementedError
            [description]
        """
        dtypes = self.dtypes
        if not dtypes or not all(dtype == dtypes[0] for dtype in dtypes):
            # No common dtype.
            return None
        return dtypes[0]

    def as_namedtuple(self) -> Tuple[Item, ...]:
        return self._namedtuple(**self.as_dict())
    
    def as_list_of_tuples(self) -> Iterable[Tuple[Item, ...]]:
        """Returns an iterable of the items in the 'batch', each item as a
        namedtuple (list of tuples).
        """
        # If one of the fields is None, then we convert it into a list of Nones,
        # so we can zip all the fields to create a list of tuples.
        field_items = [
            [items for _ in range(self.batch_size)] if items is None or items is {} else
            [item for item in items]
            for items in self.as_tuple()
        ]
        assert all([len(items) == self.batch_size for items in field_items])
        return list(itertools.starmap(self._namedtuple, zip(*field_items)))

    def as_tuple(self) -> Tuple[Item, ...]:
        """Returns a namedtuple containing the 'batched' attributes of this
        object (tuple of lists).
        """
        # TODO: Turning on the namedtuple return value by default.
        return self.as_namedtuple()
        return tuple(
            getattr(self, f.name) for f in dataclasses.fields(self) 
        )

    def as_dict(self) -> Dict[str, Item]:
        return {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) 
        }

    def to(self, *args, **kwargs):
        return type(self)(**{
            name: item.to(*args, **kwargs) if isinstance(item, (Tensor, Batch)) else item
            for name, item in self.items()
        })

    def float(self):
        return self.to(dtype=float)

    def numpy(self):
        """Returns a new Batch object of the same type, with all Tensors
        converted to numpy arrays.

        Returns
        -------
        [type]
            [description]
        """
        return type(self)(**{
            k: v.detach().cpu().numpy() if isinstance(v, (Tensor, Batch)) else v
            for k, v in self.items()
        })

    def detach(self):
        """Returns a new Batch object of the same type, with all Tensors
        detached.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors detached.
        """
        return type(self)(**{
            k: v.detach() if isinstance(v, (Tensor, Batch)) else v for k, v in self.items()
        })

    def cpu(self, **kwargs):
        """Returns a new Batch object of the same type, with all Tensors
        moved to cpu.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors moved to CPU.
        """
        return self.to(device="cpu", **kwargs)

    def cuda(self, device=None, **kwargs):
        """Returns a new Batch object of the same type, with all Tensors
        moved to cuda device.

        Returns
        -------
        Batch
            New object of the same type, but with all tensors moved to cuda.
        """
        return self.to(device=(device or "cuda"), **kwargs)

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
        if isinstance(inputs, (tuple, list)):
            from collections.abc import Sized
            if not all(isinstance(item, Sized) for item in inputs):
                # FIXME: This could either mean that this method is being passed
                # a tuple or a list of non-Batched items, or that an individual
                # field has None as a value. Hard to distinguish these two..
                return cls(*inputs)
            
                assert False, f"This should only be used on 'batched' inputs, not {inputs}.."
                
                inputs = [
                    [item] for item in inputs
                ]

            # Convert things that aren't tensors to numpy arrays.
            # Stack tensors (to preserve their 'grad' attributes, if present).
            inputs: List[Union[np.ndarray, Tensor]] = [
                items if isinstance(items, Tensor) else
                torch.stack(items) if isinstance(items[0], Tensor) else
                np.asarray(items)
                
                for items in inputs
            ]
            
            # Ndarrays with 'object' dtype aren't supported in pytorch.
            # TODO: We convert arrays with None to lists, but is this the best
            # thing to do?
            inputs = [
                array if isinstance(array, Tensor) else
                torch.as_tensor(array) if array.dtype != np.object_ else
                array.tolist()
                for array in inputs
            ]
            return cls(*inputs)

        if isinstance(inputs, Tensor):
            return cls(inputs)
        if isinstance(inputs, np.ndarray):
            return cls(torch.as_tensor(inputs))
        if isinstance(inputs, dict):
            return cls(**inputs)
        # TODO: Do we want to allow Batch objects to contain single "items" in
        # addition to batches of items?
        if isinstance(inputs, (int, float)):
            return cls(torch.as_tensor(inputs))
        raise RuntimeError(
            f"Don't know how to turn inputs {inputs} (type {type(inputs)}) "
            f"into a Batch object of type {cls}!"
        )
        # return cls(inputs)

T = TypeVar("T")

@get_slice.register(Batch)
def get_batch_slice(value: Batch, indices: Sequence[int]) -> Batch:
    return type(value)(**{
        field_name: get_slice(field_value, indices) if field_value is not None else None
        for field_name, field_value in value.as_dict().items()
    })

@set_slice.register(Batch)
def set_batch_slice(target: Batch, indices: Sequence[int], values: Tuple[T, ...]) -> None:
    for target_item, values_item in zip(target, values):
        set_slice(target_item, indices, values_item)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
