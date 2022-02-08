from typing import ClassVar, Dict, Type

from .off_policy_method import OffPolicyAlgorithm, OffPolicyMethod


class OffPolicyMethodTests:
    Method: ClassVar[Type[OffPolicyMethod]]
    Model: ClassVar[Type[OffPolicyAlgorithm]]
    debug_dataset: ClassVar[str]
    debug_kwargs: ClassVar[Dict] = {}
