import inspect
from inspect import Signature, _empty, getsourcefile
from typing import ClassVar, List, Optional, Type

import pytest
import tqdm

# from avalanche.models import MTSimpleCNN, MTSimpleMLP, SimpleCNN, SimpleMLP
from avalanche.models.utils import avalanche_forward
from avalanche.training.strategies import BaseStrategy
from torch.nn import Module

from sequoia.common.config import Config
from sequoia.conftest import xfail_param
from sequoia.settings.sl import (
    ClassIncrementalSetting,
    TaskIncrementalSLSetting,
    ContinualSLSetting,
    DiscreteTaskAgnosticSLSetting,
)
from sequoia.settings.sl.incremental.objects import Observations, Rewards
from sequoia.conftest import slow, slow_param

from .base import AvalancheMethod
from .experience import SequoiaExperience
from .patched_models import MTSimpleCNN, MTSimpleMLP, SimpleCNN, SimpleMLP


class _TestAvalancheMethod:
    Method: ClassVar[Type[AvalancheMethod]] = AvalancheMethod

    # Names of (hyper-)parameters which are allowed to have a different default value in
    # Sequoia compared to their implementations in Avalanche.
    ignored_parameter_differences: ClassVar[List[str]] = [
        "device",
        "eval_mb_size",
        "criterion",
        "train_mb_size",
        "train_epochs",
        "evaluator",
    ]

    def test_hparams_have_same_defaults_as_in_avalanche(self):
        strategy_type: Type[BaseStrategy] = self.Method.strategy_class
        method = self.Method()
        strategy_constructor: Signature = inspect.signature(strategy_type.__init__)
        strategy_init_params = strategy_constructor.parameters

        # TODO: Use the plugin constructor as the reference, rather than the Strategy
        # constructor.
        # plugin_constructor

        for parameter_name, parameter in strategy_init_params.items():
            if parameter.default is _empty:
                continue
            assert hasattr(method, parameter_name)
            method_value = getattr(method, parameter_name)
            # Ignore mismatches in some parameters, like `device`.
            if parameter_name in self.ignored_parameter_differences:
                continue

            assert method_value == parameter.default, (
                f"{self.Method.__name__} in Sequoia has different default value for "
                f"hyper-parameter '{parameter_name}' than in Avalanche: \n"
                f"\t{method_value} != {parameter.default}\n"
                f"Path to sequoia implementation: {getsourcefile(self.Method)}\n"
                f"Path to SB3 implementation: {getsourcefile(strategy_type)}\n"
            )

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type", [SimpleCNN, SimpleMLP, MTSimpleCNN, MTSimpleMLP,],
    )
    def test_short_continual_sl_setting(
        self,
        model_type: Type[Module],
        short_continual_sl_setting: ContinualSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
        results = short_continual_sl_setting.apply(method, config)
        assert 0.05 < results.average_metrics.objective

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type", [SimpleCNN, SimpleMLP, MTSimpleCNN, MTSimpleMLP,],
    )
    def test_short_discrete_task_agnostic_sl_setting(
        self,
        model_type: Type[Module],
        short_discrete_task_agnostic_sl_setting: DiscreteTaskAgnosticSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
        results = short_discrete_task_agnostic_sl_setting.apply(method, config)
        assert 0.05 < results.average_metrics.objective

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type", [SimpleCNN, SimpleMLP, MTSimpleCNN, MTSimpleMLP,],
    )
    def test_short_task_incremental_setting(
        self,
        model_type: Type[Module],
        short_task_incremental_setting: TaskIncrementalSLSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
        results = short_task_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            SimpleCNN,
            SimpleMLP,
            MTSimpleCNN,
            MTSimpleMLP,
            # xfail_param(
            #     MTSimpleCNN,
            #     reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            # ),
            # xfail_param(
            #     MTSimpleMLP,
            #     reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            # ),
        ],
    )
    def test_short_class_incremental_setting(
        self,
        model_type: Type[Module],
        short_class_incremental_setting: ClassIncrementalSetting,
        config: Config,
    ):
        method = self.Method(model=model_type, train_mb_size=10, train_epochs=1)
        results = short_class_incremental_setting.apply(method, config)
        assert 0.05 < results.average_final_performance.objective

    @slow
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "model_type",
        [
            SimpleCNN,
            SimpleMLP,
            slow_param(MTSimpleCNN),
            slow_param(MTSimpleMLP),
            # xfail_param(
            #     MTSimpleCNN,
            #     reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            # ),
            # xfail_param(
            #     MTSimpleMLP,
            #     reason="IndexError Bug inside `avalanche/models/dynamic_modules.py",
            # ),
        ],
    )
    def test_short_sl_track(
        self,
        model_type: Type[Module],
        short_sl_track_setting: ClassIncrementalSetting,
        config: Config,
    ):
        # Use the same batch size as the setting, since it's shorter than usual.
        method = self.Method(
            model=model_type,
            train_mb_size=short_sl_track_setting.batch_size,
            train_epochs=1,
        )
        results = short_sl_track_setting.apply(method, config)
        results.cl_score
        # TODO: Set up a more reasonable bound on the expected performance. For now this
        # is fine as we're just debugging: the test passes as long as there is a results
        # object that contains a non-zero online performance (meaning that the setting
        # was monitoring training performance correctly).
        assert 0 < results.average_online_performance.objective
        assert 0 < results.average_final_performance.objective


def test_warning_if_environment_to_experience_isnt_overwritten(short_sl_track_setting):
    """ When
    """
    method = AvalancheMethod()
    assert short_sl_track_setting.monitor_training_performance
    with pytest.warns(UserWarning, match="chance accuracy"):
        method.configure(short_sl_track_setting)


class MyDummyMethod(AvalancheMethod):
    def environment_to_experience(self, env, setting):
        all_observations: List[Observations] = []
        all_rewards: List[Rewards] = []

        for batch in tqdm.tqdm(env, desc="Converting environment into TensorDataset"):
            observations: Observations
            rewards: Optional[Rewards]
            if isinstance(batch, Observations):
                observations = batch
                rewards = None
            else:
                assert isinstance(batch, tuple) and len(batch) == 2
                observations, rewards = batch

            if rewards is None:
                # Need to send actions to the env before we can actually get the
                # associated Reward. Here there are (at least) three options to choose
                # from:

                # Option 1: Select action at random:
                # action = env.action_space.sample()
                # if observations.batch_size != action.shape[0]:
                #     action = action[: observations.batch_size]
                # rewards: Rewards = env.send(action)

                # Option 2: Use the current model, in 'inference' mode:
                # action = self.get_actions(observations, action_space=env.action_space)
                # rewards: Rewards = env.send(action)

                # Option 3: Train an online model:
                # NOTE: You might have to change this for your strategy. For instance,
                # currently does not take any plugins into consideration.
                self.cl_strategy.optimizer.zero_grad()

                x = observations.x.to(self.cl_strategy.device)
                task_labels = observations.task_labels
                logits = avalanche_forward(self.model, x=x, task_labels=task_labels)
                y_pred = logits.argmax(-1)
                action = self.target_setting.Actions(y_pred=y_pred)

                rewards: Rewards = env.send(action)

                y = rewards.y.to(self.cl_strategy.device)
                # Train the model:
                loss = self.cl_strategy.criterion(logits, y)
                loss.backward()
                self.cl_strategy.optimizer.step()

            all_observations.append(observations)
            all_rewards.append(rewards)

        # Stack all the observations into a single `Observations` object:
        stacked_observations: Observations = Observations.concatenate(all_observations)
        x = stacked_observations.x
        task_labels = stacked_observations.task_labels
        stacked_rewards: Rewards = Rewards.concatenate(all_rewards)
        y = stacked_rewards.y
        return SequoiaExperience(
            env=env, setting=setting, x=x, y=y, task_labels=task_labels
        )


def test_no_warning_if_environment_to_experience_is_overwritten(short_sl_track_setting):
    """ When the Method doesn't overwrite the `environment_to_experience` method, we
    raise a Warning to let the User know that they can only expect chance online
    accuracy.
    """
    method = MyDummyMethod()
    assert short_sl_track_setting.monitor_training_performance
    with pytest.warns(None) as record:
        method.configure(short_sl_track_setting)
    assert len(record) == 0
