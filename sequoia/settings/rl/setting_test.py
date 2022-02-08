""" Utilities used in tests for the RL Settings. """
from sequoia.methods import RandomBaselineMethod
from typing import List, Optional, Callable, Dict, Any
from sequoia.settings.base import Environment
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.utils.logging_utils import get_logger


logger = get_logger(__file__)


class DummyMethod(RandomBaselineMethod):
    """ Random baseline method used for debugging the (RL) settings.

    TODO: Remove the other `DummyMethod` variants, replace them with this.
    """
    def __init__(
        self,
        additional_train_wrappers: List[Callable[[Environment], Environment]] = None,
        additional_valid_wrappers: List[Callable[[Environment], Environment]] = None,
    ):
        super().__init__()
        # Wrappers to be added to the train/val environments to debug/test that the
        # setting's environments work correctly.
        self.train_env: Optional[Environment] = None
        self.valid_env: Optional[Environment] = None
        self.additional_train_wrappers = additional_train_wrappers or []
        self.additional_valid_wrappers = additional_valid_wrappers or []
        self.all_train_values = []
        self.all_valid_values = []
        self.observation_task_labels: List[Any] = []
        self.n_fit_calls = 0
        self.n_task_switches = 0
        self.received_task_ids: List[Optional[int]] = []
        self.received_while_training: List[bool] = []
        self.train_steps_per_task: List[int] = []
        self.train_episodes_per_task: List[int] = []
        self._has_been_configured_before = False

        self.changing_attributes: List[str] = []

    def configure(self, setting):
        super().configure(setting)
        if self._has_been_configured_before:
            raise RuntimeError("Can't reuse this Method across Settings for now.")
        self._has_been_configured_before = True
        # The attributes to look for changes with.
        self.changing_attributes = list(
            set().union(*[task.keys() for task in setting.train_task_schedule.values()])
        )
        self.train_env = None
        self.valid_env = None
        # Reset stuff, just so we can reuse this Method between tests maybe.
        # self.n_fit_calls = 0
        # self.train_wrappers.clear()
        # self.valid_wrappers.clear()
        # self.all_train_values.clear()
        # self.all_valid_values.clear()
        # self.observation_task_labels.clear()
        # self.n_fit_calls = 0
        # self.n_task_switches = 0
        # self.received_task_ids.clear()
        # self.received_while_training.clear()
        # self.train_steps_per_task.clear()
        # self.train_episodes_per_task.clear()

    def fit(
        self, train_env: Environment, valid_env: Environment,
    ):
        # Add wrappers, if necessary.
        for wrapper in self.additional_train_wrappers:
            train_env = wrapper(train_env)
        for wrapper in self.additional_valid_wrappers:
            valid_env = wrapper(valid_env)

        train_env = CheckAttributesWrapper(
            train_env, attributes=self.changing_attributes
        )
        valid_env = CheckAttributesWrapper(
            valid_env, attributes=self.changing_attributes
        )
        self.train_env = train_env
        self.valid_env = valid_env
        # TODO: Replace the loop below with adding soem wrappers around the train/valid envs, and
        # just delegate to super().fit (so we use the RandomBaselineMethod).
        # return super().fit(train_env, valid_env)

        episodes = 0
        val_interval = 10
        total_steps = 0
        self.train_steps_per_task.append(0)
        self.train_episodes_per_task.append(0)
        import tqdm
        train_pbar = tqdm.tqdm(desc="Fake training")
        while not train_env.is_closed():

            obs = train_env.reset()
            task_labels = obs.task_labels
            if (
                task_labels is None
                or isinstance(task_labels, int)
                or not task_labels.shape
            ):
                task_labels = [task_labels]
            self.observation_task_labels.extend(task_labels)

            done = False
            while not done and not train_env.is_closed():
                actions = train_env.action_space.sample()
                # print(train_env.current_task)
                obs, rew, done, info = train_env.step(actions)
                total_steps += 1
                self.train_steps_per_task[-1] += 1
                train_pbar.update()
                train_pbar.set_postfix({"episodes": episodes, "total steps": total_steps})

            episodes += 1
            self.train_episodes_per_task[-1] += 1

            if episodes % val_interval == 0 and not valid_env.is_closed():
                # Perform one 'validation' episode.
                obs = valid_env.reset()
                done = False
                while not done and not valid_env.is_closed():
                    actions = valid_env.action_space.sample()
                    obs, rew, done, info = valid_env.step(actions)

            if self.max_train_episodes is not None and episodes < self.max_train_episodes:
                break

        self.all_train_values.append(self.train_env.values)
        self.all_valid_values.append(self.valid_env.values)
        self.n_fit_calls += 1

    def on_task_switch(self, task_id: Optional[int] = None):
        self.n_task_switches += 1
        self.received_task_ids.append(task_id)
        self.received_while_training.append(self.training)


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
