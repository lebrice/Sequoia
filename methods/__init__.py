from typing import List, Type
from importlib import import_module
import glob
from os.path import dirname, basename, isfile, join

from settings.base import Method
AbstractMethod = Method

all_methods: List[Type[Method]] = []

def register_method(new_method: Type[Method]) -> Type[Method]:
    if new_method not in all_methods:
        all_methods.append(new_method)
    return new_method

from .baseline_method import BaselineMethod
from .random_baseline import RandomBaselineMethod


## Pretty hacky: Dynamically import all the modules defined in this folder:
modules = glob.glob(join(dirname(__file__), "*"))

all_modules: List[str] = [
    basename(f)[:-3] for f in modules
    if (isfile(f) and
    not f.endswith('__init__.py') and
    not f.endswith("_test.py"))
    ]
__all__ = all_modules
for module in all_modules:
    import_module("methods." + module)

# # TODO: We could also 'register' the methods as they are declared!
# from .stable_baselines_method import A2CMethod, PPOMethod

# TODO: (#17): Add Pl Bolts Models as Methods on IID Setting.
# from .pl_bolts_methods.cpcv2 import CPCV2Method

# print(" All methods: ", all_methods)