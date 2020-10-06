""" Defines a 'smarter' Transform class. """
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Callable, overload, Tuple, Union, Any, TypeVar, Generic, Sized

InputType = TypeVar("InputType", bound=Sized)
OutputType = TypeVar("OutputType", bound=Sized)


class Transform(Generic[InputType, OutputType]):
    """ Callable that can also tell you its impact on the shape of inputs. """

    @abstractmethod
    def __call__(self, input: InputType) -> OutputType:
        pass
    
    # @abstractmethod
    def shape_change(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """ Gives the impact this transform would have on the shape of an input.
        
        NOTE: Maybe later if some transforms create tuples, like SIMCLR, or if
        they also create labels (like dicts) or somethings, then we probably
        will have to change this.

        TODO: To push this idea even further, we could change this so it also
        accepts a gym.Space, and return a gym.Space!
        """
        # Default to saying that this transform doesn't affect the shape.
        return input_shape
        