from abc import ABC, abstractmethod
from functools import wraps
from typing import ClassVar, Dict, Generic, Optional, Type, TypeVar, Union

from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import ActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from torch import Tensor

from sequoia.utils import get_logger

logger = get_logger(__file__)

T = TypeVar("T")
Policy = TypeVar("Policy", bound=BasePolicy)
SB3Algo = TypeVar("SB3Algo", bound=BaseAlgorithm)

Wrapper = TypeVar("Wrapper", bound="PolicyWrapper")


class PolicyWrapper(BasePolicy, ABC, Generic[Policy]):
    """Base class for 'wrappers' to be applied to policies from SB3.

    This adds "hooks" into the `step()` and `zero_grad()` method of the Policy's
    optimizer.

    NOTE: Hasn't been worked on in a while, would not recommend using this unless you're
    very familiar with SB3 source code and there is no other way of doing what you want.
    """

    # Dictionary that stores the types of policies that have been 'wrapped' with
    # this mixin.
    _wrapped_classes: ClassVar[Dict[Type[T], Type[Union[T, "PolicyWrapper"]]]] = {}

    def __init__(self, *args, _already_initialized: bool = False, **kwargs):
        # When calling `EWCMixin.__init__(existing_policy)`, we don't want
        # to actually call the policy's __init__.
        if not _already_initialized:
            super().__init__(*args, **kwargs)

    @abstractmethod
    def get_loss(self: Policy) -> Union[float, Tensor]:
        """This will get called before the call to `policy.optimizer.step()`
        from within the `train` method of the algos from stable-baselines3.

        You can use this to return some kind of loss tensor to use.
        """

    def before_optimizer_step(self: Policy):
        """Called before executing `self.policy.optimizer.step()` in the training
        loop of the SB3 algos.
        """

    def after_zero_grad(self: Policy):
        """Called after `self.policy.optimizer.zero_grad()` in the training
        loop of the SB3 algos.
        """
        # Backpropagate the loss here, by default, so that any grad clipping
        # also affects the grads of the loss, for instance.
        wrapper_loss = self.get_loss()
        logger.debug(f"{type(self).__name__} loss: {wrapper_loss}")
        if isinstance(wrapper_loss, Tensor) and wrapper_loss.requires_grad:
            wrapper_loss.backward(retain_graph=True)

    @classmethod
    def wrap_policy(
        cls: Type[Wrapper], policy: Policy, **mixin_init_kwargs
    ) -> Union[Policy, Wrapper]:
        """IDEA: "Wrap" a Policy, so that every time its optimizer's `step()`
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
        if not isinstance(policy, cls):
            # Dynamically change the class of this single instance to be a subclass
            # of its current class, with the addition of the EWCMixin base class.
            policy.__class__ = cls.wrap_policy_class(type(policy))
            # 'initialize' the existing object for this mixin type.
            cls.__init__(policy, _already_initialized=True, **mixin_init_kwargs)

        assert isinstance(policy, cls)
        optimizer = policy.optimizer or policy.optimizer_class
        if optimizer is None:
            raise NotImplementedError("Need to have an optimizer instance atm")

        # 'Replace' the `policy.optimizer.step` with a function that might first
        # backpropagates the loss.
        _step = optimizer.step
        # NOTE: Setting the policy's `optimizer` attribute to a new value will
        # will actually break this.
        @wraps(optimizer.step)
        def new_optimizer_step(*args, **kwargs):
            policy.before_optimizer_step()
            return _step(*args, **kwargs)

        optimizer.step = new_optimizer_step

        _zero_grad = optimizer.zero_grad

        @wraps(optimizer.zero_grad)
        def new_zero_grad(*args, **kwargs):
            _zero_grad(*args, **kwargs)
            policy.after_zero_grad()

        optimizer.zero_grad = new_zero_grad

        return policy

    @classmethod
    def wrap_policy_class(
        cls: Type[Wrapper], policy_type: Type[Policy]
    ) -> Type[Union[Policy, Wrapper]]:
        """Add the wrapper as a base class to a policy type from SB3."""
        assert issubclass(policy_type, BasePolicy)
        if issubclass(policy_type, cls):
            # It already has the mixin, so return the class unchanged.
            return policy_type

        # Save the results so we don't create two wrappers for the same class.
        if policy_type in cls._wrapped_classes:
            return cls._wrapped_classes[policy_type]

        class WrappedPolicy(policy_type, cls):  # type: ignore
            pass

        WrappedPolicy.__name__ = policy_type.__name__ + "With" + cls.__name__
        cls._wrapped_classes[policy_type] = WrappedPolicy
        return WrappedPolicy

    @classmethod
    def wrap_algorithm(cls: Type[Wrapper], algo: SB3Algo, **wrapper_kwargs) -> SB3Algo:
        """Wrap an existing algorithm's policy using this wrapper."""
        assert isinstance(algo, BaseAlgorithm)
        if not isinstance(algo.policy, cls):
            # Dynamically change the class of this single instance to be a subclass
            # of its current class, with the addition of the EWCMixin base class.
            if algo.policy is None:
                # We want to wrap the _setup_model so the policy gets wrapped.
                # raise NotImplementedError("TODO")
                _original_setup_model = algo._setup_model

                @wraps(algo._setup_model)
                def _wrapped_setup_model(*args, **kwargs) -> None:
                    _original_setup_model(*args, **kwargs)
                    assert isinstance(algo.policy, BasePolicy)
                    algo.policy = cls.wrap_policy(algo.policy, **wrapper_kwargs)

                algo._setup_model = _wrapped_setup_model
            else:
                algo.policy = cls.wrap_policy(algo.policy, **wrapper_kwargs)
        return algo

    @classmethod
    def wrap_algorithm_class(
        cls: Type[Wrapper], algo_type: Type[SB3Algo]
    ) -> Type[Union[SB3Algo, Wrapper]]:
        """Same idea, but wraps a class of algorithm, so that its policies are
        wrapped with this mixin.
        """
        if algo_type in cls._wrapped_classes:
            return cls._wrapped_classes[algo_type]

        class WrappedAlgo(algo_type):  # type: ignore
            def __init__(self, *args, **kwargs):
                # IDEA Extract the arguments that could be used for the wrapper?
                super().__init__(*args, **kwargs)
                self.policy: Union[BasePolicy, Wrapper]

            def _setup_model(self):
                super()._setup_model()
                # TODO: Figure out a way of passing the kwargs to the policy?
                # maybe using the 'policy_kwargs' argument to the constructor?
                self.policy = cls.wrap_policy(self.policy)

            # No need to change the train loop anymore!
            # def train(self) -> None:
            #     return super().train()

            # IDEA: Redirect any failing attribute lookups to the policy?
            def __getattr__(self, attr: str):
                try:
                    return super().__getattribute__(attr)
                except AttributeError as e:
                    if hasattr(self.policy, attr):
                        return getattr(self.policy, attr)
                    raise e

            # The above would remove the need for any of these:
            # def on_task_switch(self, task_id: Optional[int]):
            #     self.policy.on_task_switch(task_id)

            # def ewc_loss(self) -> Union[float, Tensor]:
            #     return self.policy.ewc_loss()

        WrappedAlgo.__name__ = algo_type.__name__ + "With" + cls.__name__

        cls._wrapped_classes[algo_type] = WrappedAlgo
        return WrappedAlgo


from stable_baselines3 import A2C


# Either 'manually', like this:
class A2CWithEWC(A2C):
    def __init__(self, *args, ewc_coefficient: float = 1.0, ewc_p_norm: int = 2, **kwargs):
        self.ewc_coefficient = ewc_coefficient
        self.ewc_p_norm = ewc_p_norm
        super().__init__(*args, **kwargs)
        self.policy: Union[ActorCriticPolicy, EWC]

    def _setup_model(self):
        super()._setup_model()
        # Just to show that the policy was just wrapped.
        self.policy = EWC._wrap_policy(
            self.policy,
            ewc_coefficient=self.ewc_coefficient,
            ewc_p_norm=self.ewc_p_norm,
        )

    def on_task_switch(self, task_id: Optional[int]) -> None:
        self.policy.on_task_switch(task_id)


## OR automatically, like this!
# A2CWithEWC = EWC._wrap_algorithm_class(A2C)
# DQNWithEWC = EWC._wrap_algorithm_class(DQN)
# PPOWithEWC = EWC._wrap_algorithm_class(PPO)
# DDPGWithEWC = EWC._wrap_algorithm_class(DDPG)
# SACWithEWC = EWC._wrap_algorithm_class(SAC)
