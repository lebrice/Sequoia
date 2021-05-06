""" WIP: Tests for the CWRStar Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type
from .base import AvalancheMethod
from .cwr_star import CWRStarMethod
from .base_test import TestAvalancheMethod


class TestCWRStarMethod(TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = CWRStarMethod
