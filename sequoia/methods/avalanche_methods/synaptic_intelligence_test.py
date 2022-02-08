""" WIP: Tests for the SynapticIntelligence Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .base_test import _TestAvalancheMethod
from .synaptic_intelligence import SynapticIntelligenceMethod


class TestSynapticIntelligenceMethod(_TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = SynapticIntelligenceMethod
