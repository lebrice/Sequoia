from typing import ClassVar, Dict, Type

import pytest
from sequoia.common.config import Config
from sequoia.settings.rl import DiscreteTaskAgnosticRLSetting

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import DiscreteActionSpaceMethodTests
from .off_policy_method import OffPolicyAlgorithm, OffPolicyMethod


class OffPolicyMethodTests:
    Method: ClassVar[Type[OffPolicyMethod]]
    Model: ClassVar[Type[OffPolicyAlgorithm]]
    debug_dataset: ClassVar[str]
    debug_kwargs: ClassVar[Dict] = {}

