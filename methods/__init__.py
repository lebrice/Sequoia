from typing import List, Type

from settings.method_abc import MethodABC
AbstractMethod = MethodABC
from .method import Method
from .baseline import BaselineMethod
from .random_baseline import RandomBaselineMethod
from .self_supervision import SelfSupervision

# from .pl_bolts_methods.cpcv2 import CPCV2Method
# TODO: We could also 'register' the methods as they are declared!

all_methods: List[Type[MethodABC]] = [
    BaselineMethod,
    RandomBaselineMethod,
    SelfSupervision,
    # CPCV2Method, TODO: (#17): Add Pl Bolts Models as Methods on IID Setting.
    # ClassIncrementalMethod,
    # TaskIncrementalMethod,
    # SelfSupervision,
]

# print(" All methods: ", all_methods)

def register_method(new_method: Type[Method]) -> Type[Method]:
    if new_method not in all_methods:
        all_methods.append(new_method)
    return new_method
