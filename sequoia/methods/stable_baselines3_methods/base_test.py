from inspect import Parameter, Signature, getsourcefile, signature
from typing import ClassVar, Dict, Type

import pytest
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from sequoia.common.config import Config
from sequoia.conftest import monsterkong_required
from sequoia.methods.method_test import MethodTests
from sequoia.settings.base import Results
from sequoia.settings.rl import DiscreteTaskAgnosticRLSetting, IncrementalRLSetting, RLSetting

from .base import BaseAlgorithm, StableBaselines3Method

# @pytest.mark.parametrize(
#     "MethodType, AlgoType",
#     [
#         (OnPolicyMethod, OnPolicyAlgorithm),
#         (OffPolicyMethod, OffPolicyAlgorithm),
#         (A2CMethod, A2C),
#         (DDPGMethod, DDPG),
#         (PPOMethod, PPO),
#         (DQNMethod, DQN),
#         (TD3Method, TD3),
#         (SACMethod, SAC),
#     ],
# )


class StableBaselines3MethodTests(MethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = StableBaselines3Method
    Model: ClassVar[Type[BaseAlgorithm]]
    SB3_Algo: ClassVar[Type[BaseAlgorithm]]
    debug_kwargs: ClassVar[Dict] = {}

    @pytest.mark.parametrize("clear_buffers", [False, True])
    def test_clear_buffers_between_tasks(self, clear_buffers: bool, config: Config):
        setting_kwargs = dict(
            nb_tasks=2,
            train_steps_per_task=1_000,
            test_steps_per_task=1_000,
            config=config,
        )
        setting_kwargs.update(self.setting_kwargs)
        setting = DiscreteTaskAgnosticRLSetting(**setting_kwargs)
        setting.setup()
        assert setting.train_max_steps == 2_000
        assert setting.test_max_steps == 2_000
        method = self.Method(hparams=self.Model.HParams(clear_buffers_between_tasks=clear_buffers))
        method.configure(setting)
        method.fit(
            train_env=setting.train_dataloader(),
            valid_env=setting.val_dataloader(),
        )
        assert method.hparams.clear_buffers_between_tasks == clear_buffers

        # TODO: Not clear how to check the length of the replay buffer!
        length_before_task_switch = get_current_length_of_replay_buffer(method.model)

        method.on_task_switch(task_id=1)

        if clear_buffers:
            assert get_current_length_of_replay_buffer(method.model) == 0
        else:
            assert get_current_length_of_replay_buffer(method.model) == length_before_task_switch

    def test_hparams_have_same_defaults_as_in_sb3(
        self,
    ):
        hparams = self.Model.HParams()
        AlgoType = [
            cls for cls in self.Model.mro() if cls.__module__.startswith("stable_baselines3")
        ][0]
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
                f"{self.Method.__name__} in Sequoia has different default value for "
                f"hyper-parameter '{attr_name}' than in SB3: \n"
                f"\t{value_in_hparams} != {sb3_default}\n"
                f"Path to sequoia implementation: {getsourcefile(self.Method)}\n"
                f"Path to SB3 implementation: {getsourcefile(AlgoType)}\n"
            )

    @classmethod
    @pytest.fixture
    def method(cls, config: Config) -> StableBaselines3Method:
        """Fixture that returns the Method instance to use when testing/debugging."""
        return cls.Method(**cls.debug_kwargs)

    def validate_results(
        self,
        setting: RLSetting,
        method: StableBaselines3Method,
        results: RLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        # TODO: Set some 'reasonable' bounds on the performance here, depending on the
        # setting/dataset.

    def test_debug(self, method: StableBaselines3Method, setting: RLSetting, config: Config):
        results: Results = setting.apply(method, config=config)
        assert results.objective is not None
        print(results.summary())
        self.validate_results(setting=setting, method=method, results=results)


class DiscreteActionSpaceMethodTests(StableBaselines3MethodTests):
    debug_kwargs: ClassVar[Dict] = {}
    expected_debug_mean_episode_reward: ClassVar[float] = 135
    setting_kwargs: ClassVar[str] = {"dataset": "CartPole-v0"}

    @pytest.mark.timeout(120)
    @monsterkong_required
    def test_monsterkong(self):
        method = self.Method(**self.debug_kwargs)
        setting = IncrementalRLSetting(
            dataset="monsterkong",
            nb_tasks=2,
            train_steps_per_task=1_000,
            test_steps_per_task=1_000,
        )
        results: IncrementalRLSetting.Results = setting.apply(method, config=Config(debug=True))
        print(results.summary())


from functools import singledispatch

from stable_baselines3.common.buffers import RolloutBuffer


@singledispatch
def get_current_length_of_replay_buffer(algo: BaseAlgorithm) -> int:
    """Returns the current length of the replay buffer of the given Algorithm."""
    raise NotImplementedError(algo)


@get_current_length_of_replay_buffer.register
def _(algo: OffPolicyAlgorithm):
    return algo.replay_buffer.pos


@get_current_length_of_replay_buffer.register
def _(algo: OnPolicyAlgorithm):
    rollout_buffer: RolloutBuffer
    return algo.rollout_buffer.pos


class ContinuousActionSpaceMethodTests(StableBaselines3MethodTests):
    setting_kwargs: ClassVar[str] = {"dataset": "MountainCarContinuous-v0"}
