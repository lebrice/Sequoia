"""TODO: Idea: create a wrapper that accepts a 'policy' which will decide an
action to take whenever the `action` argument to the `step` method is None.

This policy should then accept the 'state' or something like that.
"""
from typing import Optional, Tuple, Callable, Any
import gym

class PolicyEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 policy: Optional[Callable[[Tuple], Any]] = None,
                 ):
        super().__init__(env)
        self.policy = policy
        self.prev_results: Optional[Tuple] = None

    def step(self, action: Optional[Any] = None, *args, **kwargs):
        if action is None:
            if self.prev_results is None:
                action = self.action_space.sample()
            elif self.policy is not None:
                action = self.policy(self.prev_results)
        self.prev_results = super().step(action)
        return self.prev_results

            

