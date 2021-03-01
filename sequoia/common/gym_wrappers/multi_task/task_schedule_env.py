""" TODO:  MultiTaskEnv that changes tasks when reaching points in time.
(nb of steps or episodes).
"""
import gym
from gym import spaces
from typing import List, Union, Dict
from .multi_task_env import MultiTaskEnv, EnvOrEnvFn


class TaskScheduleEnv(MultiTaskEnv):
    def __init__(
        self,
        envs: Union[gym.Env, List[EnvOrEnvFn]],
        env_task_ids: List[int] = None,
        step_schedule: Dict[int, int] = None,
        episode_schedule: Dict[int, int] = None,
    ):
        super().__init__(envs, env_task_ids=env_task_ids)
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
        self._steps += 1

        info["task_switch"] = False
        if self.schedule_keys_are_steps and self._steps in self.schedule:
            next_task_id = self.schedule[self._steps]
            self.switch_tasks(next_task_id)
            done = True
            info["task_switch"] = True
        return observation, reward, done, info

    def reset(self):
        self._episodes += 1
        if self.schedule_keys_are_episodes and self._episodes in self.schedule:
            next_task_id = self.schedule[self._episodes]
            self.switch_tasks(next_task_id)
        return super().reset()

    def __iter__(self):
        yield self.reset()
        for batch in super().__iter__():
            yield self.batch(batch)
