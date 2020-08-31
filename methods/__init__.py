from typing import List, Type

from .baseline import BaselineMethod
from .method import Method, MethodType
from .random_baseline import RandomBaselineMethod
from .self_supervision import SelfSupervisedMethod

# TODO: We could also 'register' the methods as they are declared!

all_methods: List[Type[Method]] = [
    BaselineMethod,
    RandomBaselineMethod,
    SelfSupervisedMethod,
]


def register_method(new_method: Type[MethodType]) -> Type[MethodType]:
    if new_method not in all_methods:
        all_methods.append(new_method)
    return new_method
