""" TODO: Implement and test SAC. """
from dataclasses import dataclass
from typing import ClassVar, Type
from simple_parsing import mutable_field

from stable_baselines3.sac import SAC

from methods import register_method

from .base import StableBaselines3Method, BaseAlgorithm, SB3BaseHParams

class SACModel(SAC):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the SAC Model. """
        # TODO: Create the fields from the SAC constructor arguments.
        pass


@register_method
class SACMethod(StableBaselines3Method):
    Model: ClassVar[Type[SACModel]] = SACModel

    hparams: SACModel.HParams = mutable_field(SACModel.HParams)