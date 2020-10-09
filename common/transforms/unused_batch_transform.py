""" Makes a Transform apply to Batch objects on top of its usual inputs.

"""
import functools
import inspect
from abc import ABC
from typing import Any, Callable, TypeVar, Union, Optional, Type, Tuple
from ..batch import Batch
from .transform import InputType, OutputType, Transform

InputBatchType = TypeVar("InputBatchType", bound=Batch)
OutputBatchType = TypeVar("OutputBatchType", bound=Batch)

class BatchTransform(Transform[Union[Batch, InputType], Union[Batch, OutputType]], ABC):
    """ (WIP): Base Class that makes a regular transform applicable to `Batch`
    objects in addition to its usual input type.

    When applied on a Transform, this it applicable on objects of type `Batch`
    by first extracting the tensors from it by calling `get_inputs_from_batch`,
    then applying the transform on those tensors, and then if the results aren't
    already `Batch` objects, this re-creates a `Batch` object for each
    result, using the `Batch` type of the input at that position.

    Transforms subclassing this need to have n_inputs == n_outputs.
    """

    def get_inputs_from_batch(self, batch_object: Batch[InputType], index_in_args: int=None) -> InputType:
        """ Retrieves the necessary inputs/tensors from a `Batch` object so that
        this transform can be applied on them.

        When this transform's `__call__` method only takes one positional arg
        (as is usually the case), `index_in_args` will be None.
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

# NOTE: Everythign below this isn't currently being used. Would need to debug it
# a bit more.

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
        # TODO: I don't think this would work if the function takes a single
        # argument which is itself a tuple and returns another tuple.
        
        new_args = []
        assert self is not None
        # Could also maybe use inspect to check the actual signature of
        # call_method above (outside of the _wrapper_call).
        only_one_pos_arg = len(args) == 1
        batch_object_types = []
        for i, arg in enumerate(args):
            batch_object_type: Optional[Type[Batch]] = None
            if isinstance(arg, Batch):
                batch_object_type = type(arg)
                arg = self.get_inputs_from_batch(arg, index_in_args=None if only_one_pos_arg else i)
            new_args.append(arg)
            batch_object_types.append(batch_object_type)

        # Actually call the method, with the new args.
        results = call_method(self, *new_args)
        
        if not isinstance(results, (tuple, list)):
            results = (results,)

        if len(results) != len(new_args):
            raise RuntimeError(
                f"BatchMethod should return same number of outputs as inputs! "
                f"(Input had {len(args)} args, output has {len(results)}, "
            )

        new_results = []
        for result, batch_object_type in zip(results, batch_object_types):
            if batch_object_type and not isinstance(result, batch_object_type):
                # convert the result back to the same type of 'Batch' object as
                # the corresponding inputs.
                if isinstance(result, (list, tuple)):
                    result = batch_object_type(*result)
                elif isinstance(result, dict):
                    result = batch_object_type(**result)
                else:
                    result = batch_object_type(result)
            new_results.append(result)
        return tuple(new_results)

                        
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

        