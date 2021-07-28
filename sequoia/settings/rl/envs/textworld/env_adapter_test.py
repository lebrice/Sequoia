""" TODO: Tests the interaction with the envs from textworld. """
from .env_adapter import TextWorldConvenienceWrapper
from .env import make_env
from pathlib import Path
import pytest


GAMES_DIR = Path("/home/fabrice/repos/TextWorld/notebooks/games")


@pytest.mark.parametrize("game_file", GAMES_DIR.glob("*.ulx"))
def test_basics(game_file: Path):
    env = make_env(game_file)
    env = TextWorldConvenienceWrapper(env)
    obs = env.reset()
    print(obs)

    for i in range(5):
        action = env.action_space.sample()
        # TODO: How about making it so that `env.action_space` is changing directly at
        # each step to reflect the applicable commands?
        # print("Action: ", env.action_to_str(action))
        print("Action (string) ", env.action_to_str(action))
        obs, rewards, done, info = env.step(action)
        # print(obs, rewards, done, info)

    # assert False, env.action_space