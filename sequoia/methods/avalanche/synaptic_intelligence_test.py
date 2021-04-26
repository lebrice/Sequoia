""" WIP: Tests for the SynapticIntelligence Method.

For now this only inherits the tests from the AvalancheMethod class.
"""
from typing import ClassVar, Type

from .base import AvalancheMethod
from .synaptic_intelligence import SynapticIntelligenceMethod
from .base_test import TestAvalancheMethod


class TestSynapticIntelligenceMethod(TestAvalancheMethod):
    Method: ClassVar[Type[AvalancheMethod]] = SynapticIntelligenceMethod
