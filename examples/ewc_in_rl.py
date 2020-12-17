""" Example of how to add a simplified regularization method to algos from
stable-baseline-3.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Type, TypeVar, Union

import gym
import torch
from stable_baselines3.a2c.policies import (ActorCriticCnnPolicy,
                                            ActorCriticPolicy)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from torch import Tensor

from sequoia.methods.stable_baselines3_methods import StableBaselines3Method
from sequoia.settings import TaskIncrementalRLSetting
from sequoia.utils import dict_intersection, get_logger

logger = get_logger(__file__)

T = TypeVar("T")
Policy = TypeVar("Policy", bound=BasePolicy)
SB3Algo = TypeVar("SB3Algo", bound=BaseAlgorithm)

Wrapper = TypeVar("Wrapper", bound="PolicyWrapper")

from sequoia.methods.stable_baselines3_methods.policy_wrapper import PolicyWrapper


class EWC(PolicyWrapper[Policy]):
    """ A Wrapper class that adds a `on_task_switch` and a `ewc_loss` method to
    an nn.Module (in this particular case, a Policy from SB3.)
    """
    def __init__(self: Policy,
                 *args,
                 ewc_coefficient: float = 1.0,
                 ewc_p_norm: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_coefficient = ewc_coefficient
        self.ewc_p_norm = ewc_p_norm

        self.previous_model_weights: Dict[str, Tensor] = {}

        self._previous_task: Optional[int] = None
        self._n_switches: int = 0

    def on_task_switch(self: Policy, task_id: Optional[int])-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        logger.info(f"On task switch called: task_id={task_id}")
        if self._previous_task is None and self._n_switches == 0 and not task_id:
            logger.info("Starting the first task, no EWC update.")
        elif task_id is None or task_id != self._previous_task:
            # NOTE: We also switch between unknown tasks.
            logger.info(f"Switching tasks: {self._previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self._previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(deepcopy({
                k: v.detach() for k, v in self.named_parameters()
            }))
        self._n_switches += 1

    def get_loss(self: Policy) -> Union[float, Tensor]:
        """ This will get called before the call to `policy.optimizer.step()`
        from within the `train` method of the algos from stable-baselines3.
        
        You can use this to return some kind of loss tensor to use.
        """
        return self.ewc_coefficient * self.ewc_loss()

    def ewc_loss(self: Policy) -> Union[float, Tensor]:
        """Gets an 'ewc-like' regularization loss.

        NOTE: This is a simplified version of EWC where the loss is the P-norm
        between the current weights and the weights as they were on the begining
        of the task.
        """
        if self._previous_task is None:
            # We're in the first task: do nothing.
            return 0.

        old_weights: Dict[str, Tensor] = self.previous_model_weights
        new_weights: Dict[str, Tensor] = dict(self.named_parameters())

        loss = 0.
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(new_w, old_w.type_as(new_w), p=self.ewc_p_norm)

        return loss


from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC



@dataclass
class EWCRLMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]]
    # Model = A2C
    # Model = DQN
    Model = PPO
    # Model = DDPG
    # Model = SAC

    ewc_coefficient: float = 1.0
    ewc_p_norm: int = 2

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        # Create the model, as usual:
        model = super().create_model(train_env, valid_env)
        # 'Wrap' the algorithm's policy with the EWC wrapper.
        model = EWC.wrap_algorithm(
            model,
            ewc_coefficient=self.ewc_coefficient,
            ewc_p_norm=self.ewc_p_norm,
        )
        return model

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting.

        If task labels are available, `task_id` will correspond to the index of
        the new task. Otherwise, if task labels aren't available, `task_id` will
        be `None`.

        todo: use this to customize how your method handles task transitions.
        """
        if self.model:
            self.model.policy.on_task_switch(task_id)


if __name__ == "__main__":
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0:      {"gravity": 10, "length": 0.5},
            1000:   {"gravity": 10, "length": 0.4},
        },
        max_steps = 2000,
    )
    method = EWCRLMethod(ewc_coefficient=0.)
    results_without_ewc = setting.apply(method)

    print(results_without_ewc)
    
    method = EWCRLMethod(ewc_coefficient=1.)
    results_with_ewc = setting.apply(method)

    print(results_without_ewc.summary())
    print(results_with_ewc.summary())
