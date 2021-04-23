""" EWC Method from Avalanche. """
from dataclasses import dataclass
from typing import ClassVar, Optional, Type

import gym
from avalanche.evaluation.metrics import (accuracy_metrics, forgetting_metrics,
                                          loss_metrics)
from avalanche.logging import InteractiveLogger
from avalanche.training import EvaluationPlugin
from avalanche.training.strategies import EWC, BaseStrategy
from sequoia.methods import register_method
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment,
                                      TaskIncrementalSetting)
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .base import AvalancheMethod


@register_method
@dataclass
class EWCMethod(AvalancheMethod, target_setting=ClassIncrementalSetting):
    """
    Elastic Weight Consolidation (EWC) strategy from Avalanche.
    See EWC plugin for details.
    This strategy does not use task identities.
    """
    strategy_class: ClassVar[Type[BaseStrategy]] = EWC

    # Hyperparameter to weigh the penalty inside the total loss. The larger the lambda,
    # the larger the regularization.
    ewc_lambda: float = 0.1  # todo: set the right value to use here.
    # `separate` to keep a separate penalty for each previous experience. `onlinesum`
    # to keep a single penalty summed over all previous tasks. `onlineweightedsum` to
    # keep a single penalty summed with a decay factor over all previous tasks.
    mode: str = "separate"
    # Used only if mode is `onlineweightedsum`. It specify the decay term of the
    # importance matrix.
    decay_factor: Optional[float] = None
    # if True, keep in memory both parameter values and importances for all previous
    # task, for all modes. If False, keep only last parameter values and importances. If
    # mode is `separate`, the value of `keep_importance_data` is set to be True.
    keep_importance_data: bool = False

    # Taking this from the ewc_mnist tutorial from avalanche repo:
    # choose some metrics and evaluation method.
    evaluator: EvaluationPlugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[InteractiveLogger()],
    )

    def configure(self, setting: ClassIncrementalSetting) -> None:
        super().configure(setting)

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> EWC:
        return super().create_cl_strategy(setting)

    def create_model(self, setting: ClassIncrementalSetting) -> Module:
        return super().create_model(setting)

    def make_optimizer(self, **kwargs) -> Optimizer:
        """ Creates the Optimizer object from the options. """
        return super().make_optimizer(**kwargs)

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        return super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        return super().get_actions(observations=observations, action_space=action_space)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        # TODO: Figure out if it makes sense to use this at test time (no real need for)
        # this at train time, except maybe in multi-task setting? Even then, not totally
        # sure.
        pass


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )
    # Create the Method, either manually or through the command-line:
    parser = ArgumentParser(__doc__)
    parser.add_arguments(EWCMethod, "method")
    args = parser.parse_args()
    method: EWCMethod = args.method

    results = setting.apply(method)
