""" TODO: Tests for the 'modified gravity' mujoco envs. """
from sequoia.conftest import mujoco_required
pytestmark = mujoco_required

from .modified_mass import ModifiedMassEnv
from gym.envs.mujoco import MujocoEnv
from typing import ClassVar, Type, Generic, TypeVar, Dict, List
from gym.wrappers import TimeLimit


EnvType = TypeVar("EnvType", bound=ModifiedMassEnv)


class ModifiedMassEnvTests:
    Environment: ClassVar[Type[EnvType]]

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

    def test_change_gravity_each_step(self):
        env: ModifiedMassEnv = self.Environment()
        max_episode_steps = 500
        n_episodes = 5

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

            body_part = self.body_names[0]
            start_mass = env.get_mass(body_part)

            while not done:
                previous_state = state
                state, reward, done, info = env.step(env.action_space.sample())
                env.render("human")
                episode_steps += 1
                total_steps += 1
                
                env.set_mass(body_part=body_part, mass=start_mass + 5 * total_steps / max_episode_steps)
                
                moved_up += (state[1] > previous_state[1])
                
                # print(f"Moving upward? {obs[1] > state[1]}")
            
            print(f"Gravity at end of episode: {env.gravity}")
            # TODO: Check that the position (in the observation) is obeying gravity?
            # if env.gravity <= 0:
            #     # Downward force, so should not have any significant preference for
            #     # moving up vs moving down.
            #     assert 0.4 <= (moved_up / max_episode_steps) <= 0.6, env.gravity
            # # if env.gravity == 0:
            # #     assert 0.5 <= (moved_up / max_episode_steps) <= 1.0
            # if env.gravity > 0:
            #     assert 0.5 <= (moved_up / max_episode_steps) <= 1.0, env.gravity
                
        assert total_steps == n_episodes * max_episode_steps
        initial_z = env.init_qpos[1]
        final_z = env.sim.data.qpos[1]
        assert initial_z == 0
        # Check that the robot is high up in the sky! :D
        assert final_z > 20

        # assert False, (env.init_qpos, env.sim.data.qpos)

    def test_task_schedule(self):
        # TODO: Reuse this test (and perhaps others from multi_task_environment_test.py)
        # but with this continual_half_cheetah instead of cartpole. 
        original = self.Environment()
        starting_mass = original.gravity
        import operator
        task_schedule = {
            10: dict(),
            20: operator.methodcaller("set_mass", torso=-12.0),
            30: operator.methodcaller("set_mass", torso=0.9),
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
                assert env.get_mass(body_part) == starting_mass
            elif 10 <= step < 20:
                assert env.get_mass(body_part) == starting_mass
            elif 20 <= step < 30:
                assert env.get_mass(body_part) == -12.0
            elif step >= 30:
                assert env.get_mass(body_part) == 0.9
        env.close()
