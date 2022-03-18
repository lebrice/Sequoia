from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import gym
from gym.envs.registration import EnvSpec, load

EnvType = TypeVar("EnvType", bound=gym.Env)
_EntryPoint = Union[str, Callable[..., gym.Env]]


class EnvVariantSpec(EnvSpec, Generic[EnvType]):
    def __init__(
        self,
        id: str,
        base_spec: EnvSpec,
        entry_point: Union[str, Callable[..., EnvType]] = None,
        reward_threshold: int = None,
        nondeterministic: bool = False,
        max_episode_steps=None,
        kwargs=None,
    ):
        super().__init__(
            id_requested=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
        )
        self.base_spec = base_spec

    def make(self, **kwargs) -> EnvType:
        return super().make(**kwargs)

    @classmethod
    def of(
        cls,
        original: EnvSpec,
        *,
        new_id: str,
        new_reward_threshold: Optional[float] = None,
        new_nondeterministic: Optional[bool] = None,
        new_max_episode_steps: Optional[int] = None,
        new_kwargs: Dict[str, Any] = None,
        new_entry_point: Union[str, Callable[..., gym.Env]] = None,
        wrappers: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
    ) -> "EnvVariantSpec":
        """Returns a new env spec which uses additional wrappers.

        NOTE: The `new_kwargs` update the current kwargs, rather than replacing them.
        """
        new_spec_kwargs = original.kwargs
        new_spec_kwargs.update(new_kwargs or {})
        # Replace the entry-point if desired:
        new_spec_entry_point: Union[str, Callable[..., EnvType]] = (
            new_entry_point or original.entry_point
        )

        new_reward_threshold = (
            new_reward_threshold if new_reward_threshold is not None else original.reward_threshold
        )
        new_nondeterministic = (
            new_nondeterministic if new_nondeterministic is not None else original.nondeterministic
        )
        new_max_episode_steps = (
            new_max_episode_steps
            if new_max_episode_steps is not None
            else original.max_episode_steps
        )

        # Add wrappers if desired.
        if wrappers:
            # Get the callable that creates the env.
            if callable(original.entry_point):
                env_fn = original.entry_point
            else:
                env_fn = load(original.entry_point)
            # @lebrice Not sure if there is a cleaner way to do this, maybe using
            # functools.reduce or functools.partial?
            def _new_entry_point(**kwargs) -> gym.Env:
                env = env_fn(**kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                return env

            new_spec_entry_point = _new_entry_point

        return cls(
            new_id,
            base_spec=original,
            entry_point=new_spec_entry_point,
            reward_threshold=new_reward_threshold,
            nondeterministic=new_nondeterministic,
            max_episode_steps=new_max_episode_steps,
            kwargs=new_spec_kwargs,
        )
