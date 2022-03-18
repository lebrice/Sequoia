""" WIP: Tests for the Naive Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .naive import NaiveMethod


class TestNaiveMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = NaiveMethod
