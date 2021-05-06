from inspect import Parameter, Signature, signature, getsourcefile
from pathlib import Path
from typing import Type

import pytest
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from . import (
    A2CMethod,
    DDPGMethod,
    DQNMethod,
    PPOMethod,
    SACMethod,
    TD3Method,
    OnPolicyMethod,
    OffPolicyMethod,
)
from .base import BaseAlgorithm, StableBaselines3Method


@pytest.mark.parametrize(
    "MethodType, AlgoType",
    [
        (OnPolicyMethod, OnPolicyAlgorithm),
        (OffPolicyMethod, OffPolicyAlgorithm),
        (A2CMethod, A2C),
        (DDPGMethod, DDPG),
        (PPOMethod, PPO),
        (DQNMethod, DQN),
        (TD3Method, TD3),
        (SACMethod, SAC),
    ],
)
def test_hparams_have_same_defaults_as_in_sb3(
    MethodType: Type[StableBaselines3Method], AlgoType: Type[BaseAlgorithm]
):
    hparams = MethodType.Model.HParams()
    sig: Signature = signature(AlgoType.__init__)

    for attr_name, value_in_hparams in hparams.to_dict().items():
        params_names = list(sig.parameters.keys())
        assert attr_name in params_names, f"Hparams has extra field {attr_name}"
        algo_constructor_parameter = sig.parameters[attr_name]
        sb3_default = algo_constructor_parameter.default
        if sb3_default is Parameter.empty:
            continue
        if attr_name in "verbose":
            continue  # ignore the default value of the 'verbose' param which we change.

        if (
            attr_name == "train_freq"
            and isinstance(sb3_default, tuple)
            and len(sb3_default) == 2
        ):
            # Convert the default of (1, "steps") to 1, since that's the format we use.
            if sb3_default[1] == "step":
                sb3_default = sb3_default[0]
            if isinstance(value_in_hparams, list):
                value_in_hparams = tuple(value_in_hparams)

        assert value_in_hparams == sb3_default, (
            f"{MethodType.__name__} in Sequoia has different default value for "
            f"hyper-parameter '{attr_name}' than in SB3: \n"
            f"\t{value_in_hparams} != {sb3_default}\n"
            f"Path to sequoia implementation: {getsourcefile(MethodType)}\n"
            f"Path to SB3 implementation: {getsourcefile(AlgoType)}\n"
        )