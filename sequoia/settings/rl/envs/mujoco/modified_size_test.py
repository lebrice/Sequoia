""" TODO: Tests for the 'modified size' mujoco envs. """
from typing import ClassVar, List, Type

import numpy as np
from gym.wrappers import TimeLimit
from sequoia.conftest import mujoco_required

pytestmark = mujoco_required

from .modified_size import ModifiedSizeEnv, get_geom_sizes


class ModifiedSizeEnvTests:
    Environment: ClassVar[Type[ModifiedSizeEnv]]

    def test_change_size_per_task(self):
        body_part = self.Environment.BODY_NAMES[0]

        nb_tasks = 2
        max_episode_steps = 200
        n_episodes = 2

        scale_factors: List[float] = [
            (0.5 + 2 * (task_id / nb_tasks)) for task_id in range(nb_tasks)
        ]
        default_tree = self.Environment().default_tree
        default_sizes: List[str] = get_geom_sizes(default_tree, body_part)

        task_envs: List[EnvType] = [
            # RenderEnvWrapper(
            TimeLimit(
                self.Environment(body_name_to_size_scale={body_part: scale_factor}),
                max_episode_steps=max_episode_steps,
            )
            # )
            for task_id, scale_factor in enumerate(scale_factors)
        ]

        for task_id, task_env in enumerate(task_envs):
            task_scale_factor = scale_factors[task_id]

            for episode in range(n_episodes):
                size = get_geom_sizes(task_env.tree, body_part)
                expected_size = [default_size * task_scale_factor for default_size in default_sizes]
                print(
                    f"default sizes: {default_sizes}, Size: {size}, "
                    f"task_scale_factor: {task_scale_factor}"
                )

                assert np.allclose(size, expected_size)

                state = task_env.reset()
                done = False
                steps = 0
                while not done:
                    obs, reward, done, info = task_env.step(task_env.action_space.sample())
                    steps += 1
                    # NOTE: Uncomment to visually inspect.
                    task_env.render("human")
            task_env.close()
