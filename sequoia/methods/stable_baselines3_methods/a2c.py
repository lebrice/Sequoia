from dataclasses import dataclass
from typing import ClassVar, Type, Union, Callable, Optional, Dict, Any
from gym import Env

import torch
from simple_parsing import choice, mutable_field
from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.a2c import ActorCriticPolicy
from stable_baselines3.common.base_class import GymEnv

from sequoia.utils import Serializable, Parseable
from sequoia.methods import register_method

from .base import StableBaselines3Method, SB3BaseHParams


class A2CModel(A2C):
    @dataclass
    class HParams(SB3BaseHParams):
        """ Hyper-parameters of the A2C Model. """
        # TODO: Create the fields from the A2C constructor arguments.
        pass


@register_method
@dataclass
class A2CMethod(StableBaselines3Method):
    """ Method that uses the DDPG model from stable-baselines3. """
    # changing the 'name' in this case here, because the default name would be
    # 'a_2_c'.
    name: ClassVar[str] = "a2c" 
    Model: ClassVar[Type[A2CModel]] = A2CModel
    
    # Hyper-parameters of the A2C model.
    hparams: A2CModel.HParams = mutable_field(A2CModel.HParams)