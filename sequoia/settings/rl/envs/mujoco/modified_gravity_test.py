""" TODO: Tests for the 'modified gravity' mujoco envs. """
from typing import ClassVar, Type, TypeVar

from gym.wrappers import TimeLimit

from sequoia.conftest import mujoco_required

pytestmark = mujoco_required

from .modified_gravity import ModifiedGravityEnv

EnvType = TypeVar("EnvType", bound=ModifiedGravityEnv)


class ModifiedGravityEnvTests:
    Environment: ClassVar[Type[EnvType]]

    # @pytest.mark.xfail(reason="The condition doesn't always work.")
    def test_change_gravity_each_step(self):
        env: ModifiedGravityEnv = self.Environment()
        max_episode_steps = 50
        n_episodes = 3

        # NOTE: Interestingly, the renderer will show
        # `env.frame_skip * max_episode_steps` frames per episode, even when
        # "Ren[d]er every frame" is set to False.
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        total_steps = 0

        for episode in range(n_episodes):
            initial_state = env.reset()
            done = False
            episode_steps = 0

            start_y = initial_state[1]
            moved_up = 0
            previous_state = initial_state
            state = initial_state
            while not done:
                previous_state = state
                state, reward, done, info = env.step(env.action_space.sample())
                env.render("human")
                episode_steps += 1
                total_steps += 1

                # decrease the gravity continually over time.
                # By the end, things should be floating.
                env.set_gravity(-10 + 5 * total_steps / max_episode_steps)
                moved_up += state[1] > previous_state[1]
                # print(f"Moving upward? {obs[1] > state[1]}")

            if episode_steps != max_episode_steps:
                print(f"Episode ended early?")

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

        assert total_steps <= n_episodes * max_episode_steps

        initial_z = env.init_qpos[1]
        final_z = env.sim.data.qpos[1]
        if env.gravity > 0:
            assert final_z > initial_z
        # TODO: These checks aren't deterministic, and only really "work" with
        # half-cheetah.
        # assert initial_z == 0
        # Check that the robot is high up in the sky! :D
        # assert final_z > 3
        # assert False, (env.init_qpos, env.sim.data.qpos)

    def test_task_schedule(self):
        # TODO: Reuse this test (and perhaps others from multi_task_environment_test.py)
        # but with this continual_half_cheetah instead of cartpole.
        original = self.Environment()
        starting_gravity = original.gravity

        task_schedule = {
            10: dict(gravity=starting_gravity),
            20: dict(gravity=-12.0),
            30: dict(gravity=0.9),
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
                assert env.gravity == starting_gravity
            elif 10 <= step < 20:
                assert env.gravity == starting_gravity
            elif 20 <= step < 30:
                assert env.gravity == -12.0
            elif step >= 30:
                assert env.gravity == 0.9
        env.close()
