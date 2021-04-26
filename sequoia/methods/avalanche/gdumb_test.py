""" WIP: Tests for the GDumb Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .gdumb import GDumbMethod
from .base_test import TestAvalancheMethod


class TestGDumbMethod(TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = GDumbMethod
