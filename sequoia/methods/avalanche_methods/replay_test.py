""" WIP: Tests for the Replay Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .replay import ReplayMethod


class TestReplayMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = ReplayMethod
