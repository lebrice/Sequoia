""" WIP: Tests for the EWC Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type, List
from .base import AvalancheMethod
from .ewc import EWCMethod
from .base_test import _TestAvalancheMethod


class TestEWCMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = EWCMethod
    ignored_parameter_differences: ClassVar[List[str]] = [
        "device",
        "eval_mb_size",
        "criterion",
        "train_mb_size",
        "train_epochs",
        "evaluator",
        "decay_factor",
    ]
