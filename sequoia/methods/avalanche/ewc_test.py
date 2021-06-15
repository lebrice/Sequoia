""" WIP: Tests for the EWC Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .ewc import EWCMethod
from .base_test import _TestAvalancheMethod


class TestEWCMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = EWCMethod
