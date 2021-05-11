""" WIP: Tests for the LwF Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .lwf import LwFMethod


class TestLwFMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = LwFMethod
