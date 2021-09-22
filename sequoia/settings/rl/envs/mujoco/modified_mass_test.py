""" TODO: Tests for the 'modified gravity' mujoco envs. """
import operator
from typing import ClassVar, List, Type
from gym.wrappers import TimeLimit
from sequoia.conftest import mujoco_required

pytestmark = mujoco_required

from gym.envs.mujoco import MujocoEnv

from .modified_mass import ModifiedMassEnv

class ModifiedMassEnvTests:
    Environment: ClassVar[Type[ModifiedMassEnv]]

    # names of the parts of the model which can be changed.
    body_names: ClassVar[List[str]]

    def test_generated_properties_change_the_actual_mass(self):
        env = self.Environment()
        for body_name in self.Environment.BODY_NAMES:
            # Get the value directly from the mujoco model.
            model_value = env.model.body_mass[env.model.body_names.index(body_name)]
            assert getattr(env, f"{body_name}_mass") == model_value
            new_value = model_value * 2
            setattr(env, f"{body_name}_mass", new_value)

            model_value = env.model.body_mass[env.model.body_names.index(body_name)]
            assert model_value == new_value

    def test_change_mass_each_step(self):
        env: ModifiedMassEnv = self.Environment()
        max_episode_steps = 200
        n_episodes = 3

        # NOTE: Interestingly, the renderer will show
        # `env.frame_skip * max_episode_steps` frames per episode, even when
        # "Ren[d]er every frame" is set to False.
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env: ModifiedMassEnv
        total_steps = 0

        for episode in range(n_episodes):
            initial_state = env.reset()
            done = False
            episode_steps = 0

            start_y = initial_state[1]
            moved_up = 0
            previous_state = initial_state
            state = initial_state

            body_part = self.Environment.BODY_NAMES[0]
            start_mass = env.get_mass(body_part)

            while not done:
                previous_state = state
                state, reward, done, info = env.step(env.action_space.sample())

                env.render("human")

                episode_steps += 1
                total_steps += 1

                env.set_mass(
                    **{body_part: start_mass + 5 * total_steps / max_episode_steps}
                )

                moved_up += state[1] > previous_state[1]
                print(f"Moving upward? {moved_up}")

        initial_z = env.init_qpos[1]
        final_z = env.sim.data.qpos[1]
        # TODO: Check that the change in mass had an impact

    def test_set_mass_with_task_schedule(self):
        body_part = "torso"
        original = self.Environment()
        starting_mass = original.get_mass("torso")
        task_schedule = {
            10: dict(),
            20: operator.methodcaller("set_mass", torso=starting_mass * 2),
            30: operator.methodcaller("set_mass", torso=starting_mass * 4),
        }
        from sequoia.common.gym_wrappers import MultiTaskEnvironment

        env = MultiTaskEnvironment(original, task_schedule=task_schedule)
        env.seed(123)
        env.reset()
        for step in range(100):
            _, _, done, _ = env.step(env.action_space.sample())
            # env.render()
            if done:
                env.reset()

            if 0 <= step < 10:
                assert env.get_mass(body_part) == starting_mass, step
            elif 10 <= step < 20:
                assert env.get_mass(body_part) == starting_mass, step
            elif 20 <= step < 30:
                assert env.get_mass(body_part) == starting_mass * 2, step
            elif step >= 30:
                assert env.get_mass(body_part) == starting_mass * 4, step
        env.close()
