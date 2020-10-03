import dataclasses
import itertools
from torch import Tensor
from dataclasses import dataclass
from typing import Sequence, Union, Any, ClassVar, Tuple, Dict, List, Sequence, Tuple
from collections import abc as collections_abc

# WIP (@lebrice): Playing around with this idea, to try and maybe use of typed
# objects for the 'Observation', the 'Action' and the 'Reward' for each kind of
# model. Might be a bit too complicated though.

@dataclass(frozen=True)
class Batch(Sequence[Union[Tensor, Any]]):
    field_names: ClassVar[Tuple[str, ...]]

    def __post_init__(self):
        type(self).field_names = [f.name for f in dataclasses.fields(self)]
        
    def __iter__(self):
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

    def as_tuple(self):
        return dataclasses.astuple(self)

    def as_dict(self):
        return dataclasses.asdict(self)

    def to(self, *args, **kwargs):
        return type(self)(
            tensor.to(*args, **kwargs) for tensor in self
        )

    @property
    def batch_size(self) -> int:
        return self[0].shape[0]

    def __getattr__(self, attr):
        print(f"Tried to get missing attr {attr} on Batch.")
        assert False, attr
        # return getattr(self._asdict(), attr)
