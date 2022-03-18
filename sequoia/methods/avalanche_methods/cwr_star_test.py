""" WIP: Tests for the CWRStar Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .cwr_star import CWRStarMethod


class TestCWRStarMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = CWRStarMethod
