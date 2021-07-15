""" TODO: Tests for the 'modified size' mujoco envs. """
from sequoia.conftest import mujoco_required
pytestmark = mujoco_required
import random
from typing import ClassVar, Dict, Generic, List, Type, TypeVar

import numpy as np
import pytest
from gym.envs.mujoco import MujocoEnv
from gym.wrappers import TimeLimit
from sequoia.common.gym_wrappers import RenderEnvWrapper
from sequoia.methods import RandomBaselineMethod
from sequoia.settings.rl.incremental import IncrementalRLSetting

from .modified_size import ModifiedSizeEnv

EnvType = TypeVar("EnvType", bound=ModifiedSizeEnv)

# NOTE: Marking a base class with xfail also affects all subclasses!
# @pytest.mark.xfail(reason="WIP")

class ModifiedSizeEnvTests:
    Environment: ClassVar[Type[EnvType]]
    
    @pytest.mark.xfail(reason="This feature isn't implemented yet.")
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
                self.Environment(body_parts=[body_part], size_scales=[scale_factor],),
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
                print(
                    f"default size: {default_size}, Size: {size}, task_scale_factor: {task_scale_factor}"
                )

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
        #     train_steps_per_task=2_000,
        #     train_wrappers=RenderEnvWrapper,
        #     test_max_steps=10_000,
        # )
        # assert setting.nb_tasks == nb_tasks

        # # NOTE: Same as above: we use a `no-op` task schedule, rather than an empty one.
        # assert not any(setting.train_task_schedule.values())
        # assert not any(setting.val_task_schedule.values())
        # assert not any(setting.test_task_schedule.values())
        # # assert not setting.train_task_schedule
        # # assert not setting.val_task_schedule
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

@pytest.mark.xfail(reason="WIP")
def test_modify_size():
    """ TODO: Use actual strings or files to check that things make sense.
    <body name="torso" pos="0 0 1.25">
      <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1" />
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide" />
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide" />
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge" />
      <geom friction="0.9" fromto="0 0 2.9 0 0 2.1" name="torso_geom" size="0.1" type="capsule" />
      <body name="thigh" pos="0 0 2.1">
        <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge" />
        <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule" />
        <body name="leg" pos="0 0 0.35">
          <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge" />
          <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule" />
          <body name="foot" pos="0.13/2 0 0.1">
            <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge" />
            <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule" />
          </body>
        </body>
      </body>
    </body>
    """
