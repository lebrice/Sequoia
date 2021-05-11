""" WIP: Tests for the GDumb Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .gdumb import GDumbMethod
from .base_test import _TestAvalancheMethod


class TestGDumbMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = GDumbMethod
