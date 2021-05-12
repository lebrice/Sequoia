""" WIP: Tests for the GEM Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .gem import GEMMethod
from .base_test import _TestAvalancheMethod


class TestGEMMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = GEMMethod
