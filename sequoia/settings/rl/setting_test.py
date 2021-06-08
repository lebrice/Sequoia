""" Utilities used in tests for the RL Settings. """
from sequoia.methods import RandomBaselineMethod
from typing import List, Optional, Callable, Dict, Any
from sequoia.settings.base import Environment
from sequoia.common.gym_wrappers import IterableWrapper


class DummyMethod(RandomBaselineMethod):
    """ Random baseline method used for debugging the settings.

    TODO: Remove the other `DummyMethod` variants, replace them with this.
    """

    def __init__(
        self,
        train_wrappers: List[Callable[[Environment], Environment]] = None,
        valid_wrappers: List[Callable[[Environment], Environment]] = None,
    ):
        super().__init__()
        # Wrappers to be added to the train/val environments to debug/test that the
        # setting's environments work correctly.
        self.train_wrappers = train_wrappers or []
        self.valid_wrappers = valid_wrappers or []
        self.train_env: Optional[Environment] = None
        self.valid_env: Optional[Environment] = None
        self.all_train_values = []
        self.all_valid_values = []
        self.observation_task_labels: List[Any] = []
        self.n_fit_calls = 0

    def configure(self, setting):
        super().configure(setting)
        self.all_train_values.clear()
        self.all_valid_values.clear()
        self.observation_task_labels.clear()
        self.n_fit_calls = 0

    def fit(
        self, train_env: Environment, valid_env: Environment,
    ):
        # Add wrappers, if necessary.
        for wrapper in self.train_wrappers:
            train_env = wrapper(train_env)
        for wrapper in self.valid_wrappers:
            valid_env = wrapper(valid_env)
        self.train_env = train_env
        self.valid_env = valid_env
        # TODO: Fix any issues with how the RandomBaselineMethod deals with
        # RL envs
        # return super().fit(train_env, valid_env)
        episodes = 0
        val_interval = 10
        
        while not train_env.is_closed() and (
            episodes < self.max_train_episodes if self.max_train_episodes else True
        ):
            obs = train_env.reset()
            task_labels = obs.task_labels
            if task_labels is None or isinstance(task_labels, int) or not task_labels.shape:
                task_labels = [task_labels]
            self.observation_task_labels.extend(task_labels)

            done = False
            while not done and not train_env.is_closed():
                actions = train_env.action_space.sample()
                # print(train_env.current_task)
                obs, rew, done, info = train_env.step(actions)

            episodes += 1

            if episodes % val_interval == 0 and not valid_env.is_closed():
                obs = valid_env.reset()
                done = False
                while not done and not valid_env.is_closed():
                    actions = valid_env.action_space.sample()
                    obs, rew, done, info = valid_env.step(actions)
        
        if hasattr(self.train_env, "values"):
            self.all_train_values.append(self.train_env.values)
        if hasattr(self.valid_env, "values"):
            self.all_valid_values.append(self.valid_env.values)
        self.n_fit_calls += 1


class CheckAttributesWrapper(IterableWrapper):
    """ Wrapper that stores the value of a given attribute at each step. """

    def __init__(self, env, attributes: List[str]):
        super().__init__(env)
        self.attributes = attributes
        self.values: Dict[int, Dict[str, Any]] = {}
        self.steps = 0

    def step(self, action):
        if self.steps not in self.values:
            self.values[self.steps] = {}
        for attribute in self.attributes:
            self.values[self.steps][attribute] = getattr(self.env, attribute)
        self.steps += 1
        return self.env.step(action)
