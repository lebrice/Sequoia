import copy
import dataclasses
import inspect
import itertools
import logging
import math
import pickle
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import Field, InitVar, dataclass, fields
from functools import singledispatch, total_ordering, wraps
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, NamedTuple, Optional,
                    Tuple, Type, TypeVar, Union, cast, overload)

import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import choice as _choice
from simple_parsing import field
from simple_parsing.helpers import Serializable, encode
from simple_parsing.helpers.serialization import register_decoding_fn

from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import compute_identity, dict_intersection, zip_dicts

from .priors import (CategoricalPrior, LogUniformPrior, NormalPrior, Prior,
                     UniformPrior)

# from orion.core.utils.format_trials import dict_to_trial
# from orion.core.worker.trial import Trial


logger = get_logger(__file__)
T = TypeVar("T")


@overload
def uniform(min: int, max: int, default: int = None, discrete: bool = True, **kwargs) -> int:
    pass


@overload
def uniform(min: float, max: float, default: float = None, **kwargs) -> float:
    pass


@overload
def uniform(min: float, max: float, default: float = None, discrete: bool = False, **kwargs) -> float:
    pass


def uniform(min: Union[int, float],
            max: Union[int, float],
            discrete: bool=False,
            default: Union[int, float]=None,
            **kwargs) -> Union[int, float]:
    # TODO: what about uniform over a "choice"?
    if default is None:
        default = (min + max) / 2
    discrete = discrete or (isinstance(min, int) and isinstance(max, int))
    if discrete:
        default = round(default)
    prior = UniformPrior(min=min, max=max, discrete=discrete, default=default)
    return hparam(
        default=default,
        prior=prior,
        **kwargs
    )


@overload
def log_uniform(min: int, max: int, discrete: bool=True, **kwargs) -> int:
    pass

@overload
def log_uniform(min: float, max: float, discrete: bool=False, **kwargs) -> float:
    pass

def log_uniform(min: Union[int,float],
                max: Union[int,float],
                discrete: bool = False,
                default: Union[int, float]=None,
                **kwargs) -> Union[int, float]:
    prior = LogUniformPrior(min=min, max=max, discrete=discrete, default=default)
    if default is None:
        log_min = math.log(min, prior.base)
        log_max = math.log(max, prior.base)
        default = math.pow(prior.base, (log_min + log_max) / 2)
        if discrete or (isinstance(min, int) and isinstance(max, int)):
            default = round(default)
    return hparam(
        default=default,
        prior=prior,
        **kwargs,
    )

loguniform = log_uniform


@wraps(_choice)
def categorical(*choices: T, default: T = None, probabilities: Union[List[float], Dict[str, float]] = None, **kwargs: Any) -> T:
    """ Marks a field as being a categorical hyper-parameter.

    This wraps the `choice` function from `simple_parsing`.

    Returns:
        T: the result of the usual `dataclasses.field()` function (a dataclass field).
    """
    metadata = kwargs.get("metadata", {})
    default_key = default
    if len(choices) == 1 and isinstance(choices[0], dict):
        choice_dict = choices[0]
        # TODO: If we use keys here, then we have to add a step in __post_init__ of the
        # dataclass holding this field, so that it gets the corresponding value from the
        # dict.
        # IDEA: Adding some kind of 'hook' to be used by simple-parsing?
        def postprocess(value):
            if value in choice_dict:
                return choice_dict[value]
            return value
        metadata["postprocessing"] = postprocess 
        if default:
            assert default in choice_dict.values()
            default_key = [k for k, v in choice_dict.items() if v == default][0]
        options = list(choice_dict.keys())
    else:
        options = list(choices)

    if isinstance(probabilities, dict):
        probabilities = [
            probabilities.get(k, 0.) for k in options
        ]

    prior = CategoricalPrior(
        choices=options,
        probabilities=probabilities,
        default_value=default_key,
    )
    metadata["prior"] = prior
    kwargs["metadata"] = metadata
    return _choice(*choices, default=default, **kwargs)


def hparam(default: T,
          *args,
          prior: Union[Type[Prior[T]], Prior[T]]=None,
          **kwargs) -> T:
    metadata = kwargs.get("metadata", {})
    min: Optional[float] = kwargs.get("min", kwargs.get("min"))
    max: Optional[float] = kwargs.get("max", kwargs.get("max"))

    if prior is None:
        assert min is not None and max is not None
        # if min and max are passed but no Prior object, assume a Uniform prior.
        prior = UniformPrior(min=min, max=max)
        metadata.update({
            "min": min,
            "max": max,
            "prior": prior,
        })

    elif isinstance(prior, type) and issubclass(prior, (UniformPrior, LogUniformPrior)):
        # use the prior as a constructor.
        assert min is not None and max is not None
        prior = prior(min=min, max=max)
    
    elif isinstance(prior, Prior):
        metadata["prior"] = prior
        if isinstance(prior, (UniformPrior, LogUniformPrior)):
            metadata.update(dict(
                min=prior.min,
                max=prior.max,
            ))
        elif isinstance(prior, (NormalPrior)):
            metadata.update(dict(
                mu=prior.mu,
                sigma=prior.sigma,
            ))

    else:
        # TODO: maybe support an arbitrary callable?
        raise RuntimeError(
            "hparam should receive either: \n"
            "- `min` and `max` kwargs, \n"
            "- `min` and `max` kwargs and a type of Prior to use, \n"
            "- a `Prior` instance."
        )

    kwargs["metadata"] = metadata
    return field(
        default=default,
        *args, **kwargs, 
    )

