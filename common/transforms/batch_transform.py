""" Makes a Transform apply to Batch objects on top of its usual inputs.

"""
import functools
import inspect
from abc import ABC
from typing import Any, Callable, TypeVar, Union
from ..batch import Batch
from .transform import InputType, OutputType, Transform


class BatchTransform(Transform[Union[Batch, InputType], OutputType], ABC):
    """ABC that marks a Transform as being able to operate on 'Batch' objects
    in addition to their usual input types.
    """
    
    def get_inputs_from_batch(self, batch_object: Batch[InputType], index_in_args: int=None) -> InputType:
        """ Retrieves the necessary inputs from a Batch object
        before the transform is applied.
        
        When this transform's `__call__` method only takes one positional arg
        (as usual), `index_in_args` will be None.
        If this transform method received more than one argument,
        `index_in_args` indicates at what index `batch_object` was
        passed.
        in its  object as position argument
        in its __call__ method, passed more than one object
        
        By default, this just fetches the first element of the batch object.
        Override this in a subclass if your transform needs more than one input.
        """
        return batch_object[0]
    
    def __init_subclass__(cls, *args, **kwargs):
        cls.__call__ = _wrap_call_method(cls.__call__)
        return cls




def make_applicable_on_batch_objects(transform: Callable[[InputType], OutputType]) -> Callable[[Union[Batch, InputType]], OutputType]:
    """ Important note: This doesn't make the transform function "batched",
    is just gives it the ability to be applied on the tensors in a Batch object.
    """
    if inspect.ismethod(transform):
        return _wrap_call_method(transform)
    if inspect.isclass(transform):
        cls = transform
        cls.__call__ = _wrap_call_method(cls.__call__)
        return cls
    elif callable(transform):
        return _wrap_call_function(transform)

def _wrap_call_method(call_method: Callable[["BatchTransform", InputType], OutputType]) -> Callable[["BatchTransform", Union[Batch, InputType]], OutputType]:
    @functools.wraps(call_method)
    def _wrapped_call_method(self: "BatchTransform", *args, **kwargs):
        assert not kwargs, f"BatchTransforms shouldn't receive keyword arguments!"
        new_args = []
        assert self is not None
        # Could also maybe use inspect to check the actual signature of
        # call_method above (outside of the _wrapper_call).
        only_one_pos_arg = len(args) == 1
        for i, arg in args:
            if isinstance(arg, Batch):
                arg = self.get_inputs_from_batch(arg, index_in_args=None if only_one_pos_arg else i)
            new_args.append(arg)
        return call_method(self, *new_args)
    return _wrapped_call_method

def _wrap_call_function(call_function: Callable[[InputType], OutputType]) -> Callable[[Union[Batch, InputType]], OutputType]:
    @functools.wraps(call_function)
    def _wrapped_call_function(self: BatchTransform[InputType, OutputType], *args, **kwargs):
        assert not kwargs, f"BatchTransforms shouldn't receive keyword arguments!"
        new_args = []
        # Could also maybe use inspect to check the actual signature of
        # call_method above (outside of the _wrapper_call).
        only_one_pos_arg = len(args) == 1
        for i, arg in args:
            if isinstance(arg, Batch):
                arg = self.get_inputs_from_batch(arg, index_in_args=None if only_one_pos_arg else i)
            new_args.append(arg)
        return call_function(self, *new_args)
    return _wrapped_call


from collections import abc as collections_abc

        