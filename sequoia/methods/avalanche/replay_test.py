""" WIP: Tests for the Replay Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .replay import ReplayMethod
from .base_test import _TestAvalancheMethod


class TestReplayMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = ReplayMethod
