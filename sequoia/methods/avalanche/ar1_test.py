""" WIP: Tests for the AR1 Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .ar1 import AR1Method
from .base_test import TestAvalancheMethod


class TestAR1Method(TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = AR1Method
