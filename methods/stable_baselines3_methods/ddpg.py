""" TODO: Implement and test DDPG. """
from dataclasses import dataclass
from typing import ClassVar, Type
from simple_parsing import mutable_field


from stable_baselines3.ddpg import DDPG

from methods import register_method

from .base import StableBaselines3Method, SB3BaseHParams


class DDPGModel(DDPG):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the DDPG Model. """
        # TODO: Create the fields from the DDPG constructor arguments.
        pass


@register_method
@dataclass
class DDPGMethod(StableBaselines3Method):
    """ Method that uses the DDPG model from stable-baselines3. """
    
    Model: ClassVar[Type[DDPGModel]] = DDPGModel

    # Hyper-parameters of the DDPG model.
    hparams: DDPGModel.HParams = mutable_field(DDPGModel.HParams)
