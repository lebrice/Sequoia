""" TODO: Implement and test DDPG. """
from typing import ClassVar, Type
from dataclasses import dataclass

from simple_parsing import mutable_field
from stable_baselines3.td3 import TD3

from methods import register_method

from .base import StableBaselines3Method, SB3BaseHParams


class TD3Model(TD3):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the TD3 model. """
        # TODO: Create the fields from the TD3 constructor arguments.


@register_method
@dataclass
class TD3Method(StableBaselines3Method):
    Model: ClassVar[Type[TD3Model]] = TD3Model
    hparams: TD3Model.HParams = mutable_field(TD3Model.HParams)
