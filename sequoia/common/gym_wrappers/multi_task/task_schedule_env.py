""" MultiTaskEnv that changes tasks when reaching points in time (nb of steps or
episodes).
"""
import gym
from gym import spaces
from typing import List, Union, Dict
from .multi_task_env import MultiTaskEnv, EnvOrEnvFn
from sequoia.utils.logging_utils import get_logger


logger = get_logger(__file__)


class TaskScheduleEnv(MultiTaskEnv):
    def __init__(
        self,
        envs: Union[gym.Env, List[EnvOrEnvFn]],
        step_schedule: Dict[int, int] = None,
        episode_schedule: Dict[int, int] = None,
    ):
        if step_schedule:
            self._use_steps = True
            self.schedule = step_schedule
        elif episode_schedule:
            self._use_steps = False
            self.schedule = episode_schedule
        else:
            raise RuntimeError(
                "Need to pass one of `step_schedule` or `episode_schedule`."
            )
        
        if isinstance(envs, gym.Env):
            # If we are given a single env, but a task schedule that will affect it,
            # then
            envs = [envs for _ in self.schedule]

        super().__init__(envs)
        self._steps: int = 0
        self._episodes: int = -1  # -1 to account for the first reset.

    @property
    def schedule_keys_are_steps(self) -> bool:
        """Returns wether the schedule is step-based (vs episode-based).

        Returns
        -------
        bool
            Wether the keys in the schedule dict are steps to transition at or episodes.
        """
        return self._use_steps

    @property
    def schedule_keys_are_episodes(self) -> bool:
        """Returns wether the schedule is episode-based (vs step-based).

        Returns
        -------
        bool
            Wether the keys in the schedule dict are episodes to transition at or steps.
        """
        return not self.schedule_keys_are_steps

    def step(self, action):
        observation, reward, done, info = super().step(action)

        info["task_switch"] = False
        if self.schedule_keys_are_steps and self._steps in self.schedule:
            task = self.schedule[self._steps]
            self.switch_tasks(task)
            done = True
            info["task_switch"] = True

        self._steps += 1
        return observation, reward, done, info

    def switch_tasks(self, new_task_index: int) -> None:
        assert 0 <= new_task_index < self.nb_tasks

        # TODO: Do we want to close envs on switching tasks? or not?
        # self.env.close()
        self._current_task_index = new_task_index
        logger.debug(f"Switching to env at index {new_task_index}")
        self.env = self.get_env(new_task_index)
        # TODO: Assuming the observations/action spaces don't change between tasks.

        if self._seeds and not self._using_live_envs:
            # Seed when creating the env, since we couldn't seed the env instance.
            self.env.seed(self._seeds[self._current_task_index])
    
    def reset(self):
        self._episodes += 1
        if self.schedule_keys_are_episodes and self._episodes in self.schedule:
            next_task_id = self.schedule[self._episodes]
            self.switch_tasks(next_task_id)
        return super().reset()
