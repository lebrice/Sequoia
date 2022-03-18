""" WIP: Tests for the GEM Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .gem import GEMMethod


class TestGEMMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = GEMMethod
