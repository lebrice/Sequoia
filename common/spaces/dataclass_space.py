""" IDEA: subclass of `gym.spaces.Dict` that considers dataclasses of a given
type as valid, and produces samples of that type when asked. 
""" 

import gym
from gym import spaces, Space
import dataclasses
from typing import Type, Dict
from collections.abc import Mapping


class DictSpace(spaces.Dict):
    def __init__(self, spaces=None, dataclass_type: Type = None, **spaces_kwargs):
        assert (spaces is None) or (not spaces_kwargs), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        assert dataclasses.is_dataclass(dataclass_type)
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, gym.spaces.Dict):
            spaces = spaces.spaces

        field_types = {
            field.name: field.type for field in dataclasses.fields(dataclass_type)
        }
        new_spaces: Dict[str, Space] = {}
        for key, subspace in spaces.items():
            # Get the type associated with that field.
            field_type = field_types[key]
            if isinstance(subspace, gym.spaces.Dict) and dataclasses.is_dataclass(field_type):
                # Also wrap any child dataclass recursively.
                subspace = DictSpace(subspace.spaces, field_type)
            new_spaces[key] = subspace

        super().__init__(new_spaces)
        self.dataclass_type = dataclass_type
        
    def contains(self, x):
        if isinstance(x, self.dataclass_type):
            return super().contains(dataclasses.asdict(x))
        if not isinstance(x, Mapping):
            return False
        if len(x) != len(self.spaces):
            return False
        for k, subspace in self.spaces.items():
            if k not in x:
                return False
            if not subspace.contains(x[k]):
                return False
        return True
    
    def sample(self):
        return self.dataclass_type(**super().sample())


def allow_dataclasses_as_dict_samples(space: spaces.Dict, dataclass_type: Type):
    """ wraps a Dict space so it recognizes dataclasses (e.g. Batch objects) as
    valid samples, and produces samples of the given type.
    """
    assert dataclasses.is_dataclass(dataclass_type)
    return DictSpace(space, dataclass_type)
    # TODO: IF we get recursion problems, we could check with `hasattr` first,
    # so that if it has `False` it means we're recursing into a child field of a
    # type we haven't yet finished wrapping.