""" TODO: Tests for the 'modified size' mujoco envs. """

from .modified_size import ModifiedSizeEnv
from gym.envs.mujoco import MujocoEnv
from typing import ClassVar, Type, Generic, TypeVar, Dict, List

EnvType = TypeVar("EnvType", bound=ModifiedSizeEnv)

from sequoia.settings.active.continual.incremental import IncrementalRLSetting
import random
from sequoia.common.gym_wrappers import RenderEnvWrapper
from gym.wrappers import TimeLimit
from sequoia.methods import RandomBaselineMethod
import numpy as np


class ModifiedSizeEnvTests:
    Environment: ClassVar[Type[EnvType]]

    def test_change_size_per_task(self):
        body_part = self.Environment.BODY_NAMES[0]

        nb_tasks = 3
        max_episode_steps = 1000
        n_episodes = 10

        scale_factors: List[float] = [
            (0.5 + 2 * (task_id / nb_tasks)) for task_id in range(nb_tasks)
        ]
        default_sizes: Dict[str, float] = self.Environment().get_size_dict()

        task_envs: List[EnvType] = [
            # RenderEnvWrapper(
                TimeLimit(
                    self.Environment(
                        body_parts=[body_part],
                        size_scales=[scale_factor],
                    ),
                    max_episode_steps=max_episode_steps,
                )
            # )
            for task_id, scale_factor in enumerate(scale_factors)
        ]

        for task_id, task_env in enumerate(task_envs):
            for episode in range(n_episodes):
                size = task_env.get_size(body_part)

                default_size = default_sizes[body_part]
                task_scale_factor = scale_factors[task_id]
                
                expected_size = default_size * task_scale_factor 
                print(f"default size: {default_size}, Size: {size}, task_scale_factor: {task_scale_factor}")
                
                assert np.allclose(size, expected_size)

                state = task_env.reset()
                done = False
                steps = 0
                while not done:
                    obs, reward, done, info = task_env.step(
                        task_env.action_space.sample()
                    )
                    steps += 1
                    task_env.render("human")
            task_env.close()

        # # TODO: This doesn't quite belong here, moreso in IncrementalRLSettingTest.

        # setting = IncrementalRLSetting(
        #     train_envs=task_envs,
        #     steps_per_task=2_000,
        #     train_wrappers=RenderEnvWrapper,
        #     test_steps=10_000,
        # )
        # assert setting.nb_tasks == nb_tasks

        # # NOTE: Same as above: we use a `no-op` task schedule, rather than an empty one.
        # assert not any(setting.train_task_schedule.values())
        # assert not any(setting.valid_task_schedule.values())
        # assert not any(setting.test_task_schedule.values())
        # # assert not setting.train_task_schedule
        # # assert not setting.valid_task_schedule
        # # assert not setting.test_task_schedule

        # method = RandomBaselineMethod()

        # results = setting.apply(method)
        # assert False, results

    # def test_change_size_at_each_step(self):
    #     env: ModifiedSizeEnv = self.Environment()
    #     raise NotImplementedError()
    # for episode in range(5):
    #     print(f"Gravity: {env.gravity}")
    #     state = env.reset()
    #     done = False
    #     steps = 0
    #     while not done:
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         env.render("human")
    #         steps += 1
    #         # decrease the gravity continually over time.
    #         env.set_gravity(-10 + (steps) * 0.01)
    # # TODO: Check that the position (in the observation) is moving upwards?
    # assert False, env.init_qpos

    # def test_change_size_with_task_schedule(self):
    #     # TODO: Reuse this test (and perhaps others from multi_task_environment_test.py)
    #     # but with this continual_half_cheetah instead of cartpole.
    #     original = self.Environment()
    #     # original = TimeLimit()

    #     starting_gravity = original.gravity
    #     import operator
    #     task_schedule = {
    #         100: operator.methodcaller("scale_size", torso=1.5),
    #         200: operator.methodcaller("scale_size", torso=0.5),
    #         300: operator.methodcaller("scale_size", torso=1.0),
    #     }
    #     from sequoia.common.gym_wrappers import MultiTaskEnvironment

    #     env = MultiTaskEnvironment(original, task_schedule=task_schedule)
    #     env.seed(123)
    #     env.reset()
    #     for step in range(400):
    #         _, _, done, _ = env.step(env.action_space.sample())
    #         env.render()
    #         if done:
    #             env.reset()

    #         if 0 <= step < 100:
    #             assert env.get_size(body_part) == starting_gravity
    #         elif 100 <= step < 200:
    #             assert env.get_size(body_part) == starting_gravity
    #         elif 200 <= step < 300:
    #             assert env.get_size(body_part) == -12.0
    #         elif step >= 300:
    #             assert env.get_size(body_part) == 0.9
    #     env.close()
