""" Methods: solutions to research problems (Settings).

Methods contain the logic related to the training of the algorithm. Methods are
encouraged to use a model to keep the networks / architecture / engineering code
separate from the training loop.

Sequoia includes a `BaseMethod`, along with an accompanying `Model`, which can be
used as a jumping-off point for new users. 
You're obviously also free to write your own method/model from scratch if you want!

The recommended way to start is by creating a new subclass of the Base
The best way to do so is to create your new model as a subclass of the `Model`,
which already has some neat capabilities, and can easily be extended/customized.

This `Model` is an instance of Pytorch-Lightning's `LightningModule` class, and can be
trained on the environments/dataloaders of Sequoia with a `pl.Trainer`, enabling all the
goodies associated with Pytorch-Lightning.

You can also easily add callbacks to measure your own metrics and such as you would in
Pytorch-Lightning.
"""
import glob
import inspect
import os
import warnings
from importlib import import_module
from os.path import basename, dirname, isfile, join
from pathlib import Path
from typing import List, Type
from os.path import abspath

import pkg_resources
from pkg_resources import EntryPoint
from typing import Dict
from setuptools import find_packages
from functools import lru_cache

from sequoia.settings.base import Method
from sequoia.utils.logging_utils import get_logger
logger = get_logger(__file__)


AbstractMethod = Method

_registered_methods: List[Type[Method]] = []


"""
TODO: IDEA: Add arguments to register_method that help configure the tests we
add the that method! E.g.:

```
@register_method(slow=True, requires_cuda=True, required_memory_gb=4)
class MyMethod(Method, target_setting=ContinualRLSetting):
    ...
```
"""


def register_method(method_class: Type[Method] = None, *, name: str = None, family: str = None) -> Type[Method]:
    """ Decorator around a method class, which is used to register the method.

    Can set the name of the method as well as the family when they are passed, and also
    adds the Method to the list of registered methods.
    """
    def _register_method(method_class: Type[Method] = None, *, name: str = None, family: str = None) -> Type[Method]:
        if name is not None:
            method_class.name = name
        if family is not None:
            method_class.family = family

        if not issubclass(method_class, Method):
            raise TypeError(
                "The `register_method` decorator should only be used on subclasses of "
                "`Method`."
            )

        if method_class not in _registered_methods:
            _registered_methods.append(method_class)

        return method_class

    # This is based on `dataclasses.dataclass`:
    def wrap(method_class: Type[Method]) -> Type[Method]:
        return _register_method(method_class, name=name, family=family)

    # See if we're being called as @register_method or @register_method().
    if method_class is None:
        # We're called with parens.
        return wrap

    # We're called as @register_method without parens.
    return wrap(method_class)


@lru_cache
def get_external_methods() -> Dict[str, Type[Method]]:
    """ Returns a dictionary of the Methods defined outside of Sequoia.

    Packages outside of Sequoia can register methods by putting a `Method` entry-point
    in their setup.py, like so:

    ```python
    # (inside <some_package_dir>/setup.py)

    setup(
        name="my_package",
        packages=setuptools.find_packages(include=["cn_dpm*"])
        ...
        entry_points={
            "Method": [
                "foo_method = my_package.my_methods.foo_method:FooMethod",
                "bar_method = my_package.my_methods.bar_method:BarMethod",
            ],
        },
    )
    ```

    Compared with using the `@register_method` decorator, this has the benefit that the
    module containing the Method does not need to be imported/"live" for the method to
    be available. This is very relevant when using Sequoia through the command-line, for
    instance, since Sequoia would have no way of knowing what other methods are
    available:

    ```console
    sequoia setting foo_setting method foo_method
    ```
    """
    methods: Dict[str, Type[Method]] = {}
    for entry_point in pkg_resources.iter_entry_points("Method"):
        entry_point: EntryPoint
        try:
            method_class = entry_point.load()
        except Exception as exc:
            logger.error(
                f"Unable to load external Method: '{entry_point.name}', from package "
                f"{entry_point.dist.project_name}, version={entry_point.dist.version}: "
                f"{exc}"
            )
        else:
            logger.debug(
                f"Imported an external Method: '{entry_point.name}', from package "
                f"{entry_point.dist.project_name}, (version = {entry_point.dist.version})."
            )
            methods[entry_point.name] = method_class
    return methods


from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.methods.base_method import BaseMethod, BaseModel
# Keeping a pointer to the old name, just to help with backward-compatibility a bit.
BaselineMethod = BaseMethod

from sequoia.methods.pnn import PnnMethod
from sequoia.methods.experience_replay import ExperienceReplayMethod
from sequoia.methods.hat import HatMethod
from sequoia.methods.ewc_method import EwcMethod

# TODO: Eventually these could become external repos, with their own tests / etc, based
# on a 'cookiecutter' repo of some sort. This would make it easier to maintain and to
# delegate work!

# IDEA: Could also do the same for the datasets somehow? Like have an extendable
# `sequoia.datasets` cookiecutter repo? How would that work with Settings?
# Assumption + Assumption -> Assumption (combined)
# Setting := fn(dataset, **kwargs) -> Callable[[Method], Results]


try:
    from sequoia.methods.avalanche import *
except ImportError:
    pass

try:
    from sequoia.methods.stable_baselines3_methods import *
except ImportError:
    pass


try:
    from sequoia.methods.pl_bolts_methods import *
except ImportError:
    pass



def add_external_methods(all_methods: List[Type[Method]]) -> List[Type[Method]]:
    for name, method_class in get_external_methods().items():
        if method_class not in all_methods:
            logger.debug(f"Adding method {name} from external package.")
            all_methods.append(method_class)
    return all_methods


def get_all_methods() -> List[Type[Method]]:
    # This may change over time, and includes ALL subclasses of 'Method'.
    # methods = Method.__subclasses__()
    # This includes all registered methods, e.g. not any base classes.
    methods = _registered_methods
    methods = add_external_methods(methods)  # This won't.
    methods = list(set(methods))
    return list(sorted(methods, key=lambda method: method.get_full_name()))
