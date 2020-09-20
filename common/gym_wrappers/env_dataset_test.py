from .env_dataset import EnvDataset
import gym

from gym.spaces import Discrete


class DummyEnvironment(gym.Env):
    """ Dummy environment for testing.
    
    The reward is how close to the target value the state (a counter) is. The
    actions are:
    0:  keep the counter the same.
    1:  Increment the counter.
    2:  Decrement the counter.
    """
    def __init__(self, start: int = 0, max_value: int = 10):
        self.max_value = max_value
        self.i = start
        self.reward_range = (0, max_value)
        self.action_space = Discrete(n=3)
        self.observation_space = Discrete(n=max_value)

        self.target = max_value // 2

        self.done: bool = False

    def step(self, action: int):
        # The action modifies the state, producing a new state, and you get the
        # reward associated with that transition.
        if action == 1:
            self.i += 1
        elif action == 2:
            self.i -= 1
        self.i %= self.max_value
        done = (self.i == self.target)
        reward = abs(self.i - self.target)
        print(self.i, reward, done, action)
        return self.i, reward, done, {}

    def reset(self):
        self.i = 0

def test_step_normally_works_fine():
    env = DummyEnvironment()
    env = EnvIterator(env)
    env.reset()
    env.seed(123)

    obs, reward, done, info = env.step(0)
    assert (obs, reward, done, info) == (0, 5, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (1, 4, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (2, 3, False, {})
    obs, reward, done, info = env.step(2)
    assert (obs, reward, done, info) == (1, 4, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (2, 3, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (3, 2, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (4, 1, False, {})
    
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (5, 0, True, {})

    env.reset()
    obs, reward, done, info = env.step(0)
    assert (obs, reward, done, info) == (0, 5, False, {})

def test_raise_error_when_missing_action():
    env = DummyEnvironment()
    env = EnvIterator(env)
    env.reset()
    env.seed(123)

    for i, batch in zip(range(4), env):
        assert False, batch
    env.close()

    # env = GymDataLoader(
    #     "CartPole-v0",
    #     batch_size=10,
    #     observe_pixels=False,
    #     random_actions_when_missing=False,
    # )
    # env.reset()
    # with pytest.raises(RuntimeError):
    #     for obs_batch in take(env, 5):
    #         # raises an error after the first iteration, as it didn't receive an action. 
    #         pass

    