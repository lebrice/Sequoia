""" Example of how to add a simplified regularization method to algos from
stable-baseline-3.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Type, TypeVar, Union

import gym
import torch
from sequoia.methods import register_method
from sequoia.methods.stable_baselines3_methods import StableBaselines3Method
from sequoia.methods.stable_baselines3_methods.policy_wrapper import \
    PolicyWrapper
from sequoia.settings import TaskIncrementalRLSetting
from sequoia.utils import dict_intersection, get_logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from torch import Tensor

logger = get_logger(__file__)

Policy = TypeVar("Policy", bound=BasePolicy)


class EWC(PolicyWrapper[Policy]):
    """ A Wrapper class that adds a `on_task_switch` and a `ewc_loss` method to
    an nn.Module (in this particular case, a Policy from SB3.)
    
    By subclassing PolicyWrapper, this is able to leverage some 'hooks' into the
    optimizer of the policy. 
    """
    def __init__(self: Policy,
                 *args,
                 reg_coefficient: float = 1.0,
                 ewc_p_norm: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_coefficient = reg_coefficient
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
        return self.reg_coefficient * self.ewc_loss()
    
    def after_zero_grad(self: Policy):
        """ Called after `self.policy.optimizer.zero_grad()` in the training 
        loop of the SB3 algos.
        """
        # Backpropagate the loss here, by default, so that any grad clipping
        # also affects the grads of the loss, for instance.
        wrapper_loss = self.get_loss()
        if isinstance(wrapper_loss, Tensor) and wrapper_loss != 0. and wrapper_loss.requires_grad:
            logger.info(f"{type(self).__name__} loss: {wrapper_loss.item()}")
            wrapper_loss.backward(retain_graph=True)
    
    def before_optimizer_step(self: Policy):
        """ Called before `self.policy.optimizer.step()` in the training 
        loop of the SB3 algos.
        """

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


from sequoia.methods.stable_baselines3_methods import (A2CModel, DDPGModel,
                                                       DQNModel, PPOModel,
                                                       SACModel, TD3Model)


@register_method
@dataclass
class ExampleRegularizationMethod(StableBaselines3Method):
    Model: ClassVar[Type[BaseAlgorithm]]

    # You could use any of these 'backbones' from SB3:
    # Model = A2CModel  # Works great! (fastest)
    Model = PPOModel  # Works great! (somewhat fast)
    # Model = SACModel  # Works (seems to be quite a bit slower).
    
    # These two don't yet work, they have the same error, which seems to be
    # related to the action space being Discrete:
    #     stable_baselines3/td3/td3.py", line 143, in train
    #     noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
    # RuntimeError: "normal_kernel_cuda" not implemented for 'Long'
    # Model = TD3Model  # TODO
    # Model = DDPGModel  # TODO
    # Model = DQNModel  # Doesn't work: predictions have more than one value?!

    
    
    # Coefficient for the EWC-like loss.
    reg_coefficient: float = 1.0
    # norm of the 'distance' used in the ewc-like loss above.
    ewc_p_norm: int = 2

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        # Create the model, as usual:
        model = super().create_model(train_env, valid_env)
        # 'Wrap' the algorithm's policy with the EWC wrapper.
        model = EWC.wrap_algorithm(
            model,
            reg_coefficient=self.reg_coefficient,
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
            0:      {"gravity": 10, "length": 0.3},
            1000:   {"gravity": 10, "length": 0.5}, # second task is 'easier' than the first one.
        },
        max_steps = 2000,
    )
    method = ExampleRegularizationMethod(reg_coefficient=0.)
    results_without_reg = setting.apply(method)

    method = ExampleRegularizationMethod(reg_coefficient=1e-3)
    results_with_reg = setting.apply(method)
    print("-" * 40)
    print("WITHOUT EWC ")
    print(results_without_reg.summary())
    print(f"With EWC (coefficient={method.reg_coefficient}):")
    print(results_with_reg.summary())
