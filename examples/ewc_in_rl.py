from abc import ABC
from torch import nn, Tensor
from typing import Union, Optional, Type
import torch
from sequoia.utils import get_logger, dict_intersection
from copy import deepcopy

from stable_baselines3.common.policies import BasePolicy

from stable_baselines3.a2c.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from typing import TypeVar
from functools import lru_cache
from functools import wraps
from typing import Dict

from typing import ClassVar, Type

logger = get_logger(__file__)

Mixin = TypeVar("Mixin", bound="EWCMixin")
Policy = TypeVar("Policy", bound=BasePolicy)

from stable_baselines3.a2c import A2C
from stable_baselines3.common import logger as sb3_logger

from sequoia.methods.stable_baselines3_methods.a2c import A2CMethod, A2CModel
from stable_baselines3.common.utils import explained_variance
from gym import spaces
import torch as th
from torch.nn import functional as F
from inspect import isclass

class EWCMixin(nn.Module, ABC):
    """ A Mixin class that adds a `on_task_switch` and a `ewc_loss` method to
    any nn.Module.
    """
    # Dictionary that stores the types of policies that have been 'wrapped' with
    # this mixin.
    _wrapped_classes: ClassVar[Dict[Type[BasePolicy],
                                    Type[Union[BasePolicy, Mixin]]]] = {}
    
    def __init__(self,
                 *args,
                 ewc_coefficient: float = 1.0,
                 ewc_p_norm: int = 2,
                 _already_initialized: bool = False,
                 **kwargs):
        if not _already_initialized:
            # When calling EWCMixin.__init__(existing_policy), we don't want to
            # actually call the policy's __init__.
            super().__init__(*args, **kwargs)
        self.ewc_coefficient = ewc_coefficient
        self.ewc_p_norm = ewc_p_norm

        self.previous_model_weights: Dict[str, Tensor] = {}

        self._previous_task: Optional[int] = None
        self._n_switches: int = 0

    def on_task_switch(self, task_id: Optional[int])-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        if self._previous_task is None and self._n_switches == 0:
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

    def ewc_loss(self) -> Union[float, Tensor]:
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
        
        # assert False, (loss, self.previous_model_weights.keys(), new_weights.keys())
        return self.ewc_coefficient * loss

    @classmethod
    def _wrap_class(cls: Type[Mixin], policy_type: Type[Policy]) -> Type[Union[Policy, Mixin]]:
        """ Add the EWCMixin base class to a policy type from SB3. """
        assert issubclass(policy_type, BasePolicy)
        if issubclass(policy_type, cls):
            # It already has the mixin, so return the class unchanged.
            return policy_type

        # Save the results so we don't create two wrappers for the same class. 
        if policy_type in cls._wrapped_classes:
            return cls._wrapped_classes[policy_type]
        
        class PolicyWithEWC(policy_type, cls):  # type: ignore
            pass

        PolicyWithEWC.__name__ = policy_type.__name__ + "WithEWC"
        cls._wrapped_classes[policy_type] = PolicyWithEWC
        return PolicyWithEWC
    
    @classmethod
    def _wrap(cls: Type[Mixin], policy: Policy, **mixin_init_kwargs) -> Union[Policy, Mixin]:
        """ IDEA: "Wrap" a Policy, so that every time its optimizer's `step()`
        method gets called, it actually first backpropagates an EWC loss.

        Parameters
        ----------
        policy : Policy
            [description]

        Returns
        -------
        Union[Policy, EWCMixin]
            [description]
        """
        assert isinstance(policy, BasePolicy)
        if not isinstance(policy, EWCMixin):
            # Dynamically change the class of this single instance to be a subclass
            # of its current class, with the addition of the EWCMixin base class. 
            policy.__class__ = cls._wrap_class(type(policy))
            # 'initialize' the existing object for this mixin type.
            cls.__init__(policy, _already_initialized=True, **mixin_init_kwargs)

        assert isinstance(policy, EWCMixin)

        optimzier = getattr(policy, "optimizer")
        # 'Replace' the `policy.optimizer.step` with a function that first
        # backpropagates the EWC loss.
        _optimizer_step = policy.optimizer.step
        # NOTE: Changing the policy's optimizer will actually break this. 
        @wraps(policy.optimizer.step)
        def new_optimizer_step(*args, **kwargs):
            ewc_loss = policy.ewc_loss()
            logger.info(f"EWC loss: {ewc_loss}")
            if isinstance(ewc_loss, Tensor) and ewc_loss.requires_grad:
                ewc_loss.backward()
            return _optimizer_step(*args, **kwargs)

        policy.optimizer.step = new_optimizer_step
        return policy

# Dict that stores the 'wrapped' class for each 
wrapped_policy_classes: Dict[Type[BasePolicy], Type[Union[BasePolicy, EWCMixin]]] = {
    
}


# Cache so we don't create only one wrapper per orginal policy class.
@lru_cache(maxsize=None)
def add_ewc_to_policy(policy_type: Type[Policy]) -> Type[Union[Policy, EWCMixin]]:
    """ Add the EWCMixin base class to a policy class from stable-baselines3."""
    assert issubclass(policy_type, BasePolicy)
    if issubclass(policy_type, EWCMixin):
        # It already has the mixin, so return the class unchanged.
        return policy_type

    class PolicyWithEWC(policy_type, EWCMixin):  # type: ignore
        pass

    PolicyWithEWC.__name__ = policy_type.__name__ + "WithEWC"
    return PolicyWithEWC


## IDEA: How about we 

# def wrap_policy(policy: Policy) -> Union[Policy, EWCMixin]:
    
##




# class CustomA2CPolicy(ActorCriticPolicy, EWCMixin):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

CustomActorCriticPolicy = add_ewc_to_policy(ActorCriticPolicy)
CustomActorCriticCnnPolicy = add_ewc_to_policy(ActorCriticPolicy)

class A2CWithEWC(A2CModel):
    def __init__(self, policy, *args, **kwargs):
        # if policy == "MlpPolicy":
        #     policy = CustomActorCriticPolicy
        # elif policy == "CnnPolicy":
        #     policy = CustomActorCriticCnnPolicy
        # elif isclass(policy) and issubclass(policy, BasePolicy):
        #     policy = add_ewc_to_policy(policy)
        # else:
        #     assert False, f"Unssuported policy: {policy}"
        #     policy = CustomActorCriticPolicy

        super().__init__(policy, *args, **kwargs)
        self.policy: Union[BasePolicy, EWCMixin] = EWCMixin._wrap(self.policy)
        # self.policy: CustomActorCriticPolicy

    def train(self) -> None:
        
        return super().train()
        
        
        # # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        # # This will only loop once (get all data in one go)
        # for rollout_data in self.rollout_buffer.get(batch_size=None):

        #     actions = rollout_data.actions
        #     if isinstance(self.action_space, spaces.Discrete):
        #         # Convert discrete action from float to long
        #         actions = actions.long().flatten()

        #     # TODO: avoid second computation of everything because of the gradient
        #     values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        #     values = values.flatten()

        #     # Normalize advantage (not present in the original implementation)
        #     advantages = rollout_data.advantages
        #     if self.normalize_advantage:
        #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #     # Policy gradient loss
        #     policy_loss = -(advantages * log_prob).mean()

        #     # Value loss using the TD(gae_lambda) target
        #     value_loss = F.mse_loss(rollout_data.returns, values)

        #     # Entropy loss favor exploration
        #     if entropy is None:
        #         # Approximate entropy when no analytical form
        #         entropy_loss = -th.mean(-log_prob)
        #     else:
        #         entropy_loss = -th.mean(entropy)

        #     loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        #     # Optimization step
        #     self.policy.optimizer.zero_grad()

        #     ## ADDITIONAL LOSSES:
        #     # loss += self.additional_losses(rollout_data, log_probs)
        #     ewc_loss = self.policy.ewc_loss()
        #     loss += ewc_loss
        #     ##

        #     loss.backward()

        #     # Clip grad norm
        #     th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        #     self.policy.optimizer.step()
        #     self.policy.optimizer.step()
        #     # under the hood:
        #     # self.ewc_loss().backward()
        #     # self.optimizer.step()

        # explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # self._n_updates += 1
        # # logger.info(f"train/n_updates: {self._n_updates}")
        # # logger.info(f"train/explained_variance: {explained_var}")
        # # logger.info(f"train/entropy_loss: {entropy_loss.item()}")
        # # logger.info(f"train/policy_loss: {policy_loss.item()}")
        # # logger.info(f"train/value_loss: {value_loss.item()}")
        # # NEW:
        # logger.info(f"train/ewc_loss: {ewc_loss}")

        # sb3_logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # sb3_logger.record("train/explained_variance", explained_var)
        # sb3_logger.record("train/entropy_loss", entropy_loss.item())
        # sb3_logger.record("train/policy_loss", policy_loss.item())
        # sb3_logger.record("train/value_loss", value_loss.item())
        # if hasattr(self.policy, "log_std"):
        #     sb3_logger.record("train/std", th.exp(self.policy.log_std).mean().item())


class MyA2CMethod(A2CMethod):
    Model: ClassVar[Type[A2CWithEWC]] = A2CWithEWC

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: A2CWithEWC
    
    def create_model(self, train_env, valid_env):
        model = super().create_model(train_env, valid_env)
        model.policy.on_task_switch(0)
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




from sequoia.settings import TaskIncrementalRLSetting

    
if __name__ == "__main__":
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0:      {"gravity": 10, "length": 0.2},
            1000:   {"gravity": 100, "length": 1.2},
        },
        max_steps = 2000,
    )
    method = MyA2CMethod()
    results = setting.apply(method)
    print(results)
    