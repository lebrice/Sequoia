"""Utility functions used to manipulate generators.
"""
from collections import abc
from collections.abc import Container, Iterable, Sequence, Sized
from typing import (Any, Callable, Generator, Generic, List, Sequence, TypeVar,
                    Union)
import sys; sys.path.extend([".", ".."])

from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

from setups.base import ActionType, EnvironmentBase, ObservationType, RewardType

logger = get_logger(__file__)
T = TypeVar("T")


class BatchedAccess(Generic[T]):
    """ Weird, cool class that 'batches' attribute accesses and method calls.
    
    It also batches the results of attribute accesses!

    >>> batched_list = BatchedAccess([1, 2, 3], [4, 5, 6])
    >>> batched_list.append([0, 0])
    >>> batched_list
    BatchedAccess([1, 2, 3, 0], [4, 5, 6, 0])
    >>> batched_list.pop(0)
    [1, 4]
    >>> zipped_strings = BatchedAccess("a", "B")
    >>> zipped_strings
    BatchedAccess('a', 'B')
    >>> zipped_strings.lower()
    ['a', 'b']
    >>> zipped_strings.upper()
    ['A', 'B']
    """
    
    def __init__(self, *source: T):
        self._source = list(source)

        assert self._source
        if isinstance(self._source[0], Container):
            self._is_container = True
            logger.debug(f"Wrapping a Container class!")

    def __str__(self) -> str:
        _str = self.__getattr__("__str__")
        return "BatchedAccess(" + ", ".join(_str()) + ")"

    def __repr__(self) -> str:
        _repr = self.__getattr__("__repr__")
        return "BatchedAccess(" + ", ".join(_repr()) + ")"

    def __getattr__(self, attr: str):
        logger.debug(f"Getting attribute {attr}")
        attributes = [
            getattr(source, attr) for source in self._source
        ]
        n = len(attributes)
        # If the 'attributes' are methods:
        if all(map(callable, attributes)):
            def batched_method(*args, **kwargs):
                # Split the args and kwargs for each method.
                if args:
                    logger.debug(f"Args to batched method {attr}: {args}")
                    # If the method arguments aren't batched, we replicate them.
                    args = list(self._replicate_if_needed(arg) for arg in args)
                    args: List[Tuple] = list(zip(*args))
                    logger.debug(f"args after: {args}")

                if kwargs:
                    # print(f"Kwargs to batch method {attr}: {kwargs}")
                    kwargs: List[Dict] = [{
                            k: v[i] for k, v in kwargs.items()
                        } for i in range(n)
                    ]
                results = [
                    attribute(
                        *(args[i] if args else []),
                        **(kwargs[i] if kwargs else {})
                    ) for i, attribute in enumerate(attributes)
                ]
                logger.debug(f"Results of batched method {attr} call: {results}")
                if any(result is not None for result in results):
                    return results
            return batched_method
        else:
            return BatchedAccess(attributes)

    def __setattribute__(self, attr: str, value: Union[Sequence, Any]) -> None:
        if isinstance(value, Sequence):
            for gen, val in zip(self._source, value):
                setattr(gen, attr, val)
        else:
            # Set the value on all the items.
            # TODO: Might be a bit dangerous.
            for gen in self._source:
                setattr(gen, attr, value)

    def __getitem__(self, index):
        getitem = self.__getattr__("__getitem__")
        n = len(self._source)
        index = self._replicate_if_needed(index)
        return getitem(index)

    def __setitem__(self, index, value) -> None:
        setitem = self.__getattr__("__setitem__")
        
        n = len(self._source)
        index = self._replicate_if_needed(index)
        value = self._replicate_if_needed(value)
        setitem(index, value)

    def _replicate_if_needed(self, val: Union[Sequence[T], T]) -> Sequence[Union[T, Sequence[T]]]:
        n = len(self._source)
        if isinstance(val, abc.Sequence) and len(val) == n:
            return val
        else:
            return [val] * n


class BatchEnvironments(BatchedAccess, EnvironmentBase[List[ObservationType], List[ActionType], List[RewardType]], IterableDataset):
    """TODO: Trying to create a 'batched' version of a Generator.
    """
    def __init__(self, *generators: EnvironmentBase[ObservationType, ActionType, RewardType],
                        collate_fn: Callable[[List], Any]=None):
        self.generators = generators
        self.collate_fn = collate_fn
        super().__init__(*self.generators)
    
    def __next__(self) -> List[ObservationType]:
        values = list(next(gen) for gen in self.generators)
        if self.collate_fn:
            return self.collate_fn(values)
        return values
    
    def __iter__(self) -> Generator[List[ObservationType], List[ActionType], None]:
        # iterators = (
        #     iter(g) for g in self.generators
        # )
        while True:
            actions = yield next(self)
            if actions is not None:
                raise RuntimeError("Wait, someone used the iterator as a generator instead of sending the value with `send`!")
    
    def send(self, actions: List[ActionType]) -> List[RewardType]:
        if actions is not None:
            assert len(actions) == len(self.generators)
            self.action = actions
        return [
            gen.send(action) for gen, action in zip(self.generators, actions)
        ]

    def close(self):
        for env in self.generators:
            env.close()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

