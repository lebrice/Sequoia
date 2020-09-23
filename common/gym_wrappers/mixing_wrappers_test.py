"""
Tests that check that combining wrappers works fine in combination.
"""

from common.gym_wrappers import BatchEnv, EnvDataset, MultiTaskEnvironment, PixelStateWrapper


def make_env_factory(env_name: str = "CartPole-v0",
                     batch_size: int = 10,
                     observe_pixels: bool = False,
                     ) -> Callable[[], Union[BatchEnv, EnvDataset]]:
    def make_base_env():
        env = gym.make(env_name)
        return env

    def env_factory():
        # TODO: Figure out the right ordering to use for the wrappers.
        env = BatchEnv(env_factory=make_base_env, batch_size=batch_size)
        if observe_pixels:
            env = PixelStateWrapper(env)
        env = EnvDataset(env)

        # Where do we fit in this wrapper?
        # env = MultiTaskEnvironment(env)
        return env
    return env_factory


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, 10, 32])
def test_wrappers(env_name: str, batch_size: int):
    env_factory = make_env_factory(env_name=env_name, batch_size=batch_size)
    env: Union[BatchEnv, EnvDataset] = env_factory()
    start_state = env.reset()
    assert start_state.shape == (batch_size, 4)

    for i in range(10):
        action = env.random_actions()
        assert action.shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (batch_size, 4)
        assert reward.shape == (batch_size,)

from conftest import xfail_param
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [
    xfail_param(1, "TODO: The SubProcVecEnv from openai baselines actually "
                   "'flattens' the observations, so that '1' dimension gets "
                   "destroyed. atm."),
    5,
    10,
    32,
])
def test_wrappers_with_pixels(env_name: str, batch_size: int):
    env_factory = make_env_factory(env_name=env_name, batch_size=batch_size, observe_pixels=True)
    env: Union[BatchEnv, EnvDataset] = env_factory()
    start_state = env.reset()
    expected_state_shape = (batch_size, 400, 600, 3)
    assert start_state.shape == expected_state_shape

    for i in range(10):
        action = env.random_actions()
        assert action.shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == expected_state_shape
        assert reward.shape == (batch_size,)
