import os
from pathlib import Path
from typing import List, Union

import gym
import numpy as np
from sequoia.methods import Method
from sequoia.settings.rl import RLSetting

import textworld
from textworld import EnvInfos, Game
from textworld.gym import Agent
from textworld.gym import register_game, register_games
from textworld.gym.envs import TextworldGymEnv, TextworldBatchGymEnv
from textworld.gym.spaces import Char, Word
from textworld.text_utils import extract_vocab, extract_vocab_from_gamefiles


def _get_vobab(gamefiles: Union[str, List[str]]) -> List[str]:
    # file_paths: list[Path] = list(map(Path, files))
    # for game_file in file_paths:
    #     jsonfile = game_file.with_suffix(".json")
    #     assert jsonfile.exists()
    #     game = Game.load(jsonfile)
    # assert False, game.kb.inform7_commands
    files = [gamefiles] if not isinstance(gamefiles, list) else gamefiles
    vocab = extract_vocab_from_gamefiles(map(str, files))
    vocab = sorted(vocab)  # Sorting the vocabulary, optional.
    return vocab


def get_action_space(gamefiles: Union[str, List[str]]) -> Word:
    return Word(max_length=8, vocab=_get_vobab(gamefiles))


def get_observation_space(gamefiles: Union[str, List[str]]) -> Word:
    return Word(max_length=200, vocab=_get_vobab(gamefiles))


def make_env(path: Path, max_episode_steps: int=100) -> TextworldGymEnv:
    # Get all the information we can:
    infos_to_request = EnvInfos(
        description=True, inventory=True, admissible_commands=True, won=True, lost=True,
    )

    infos_to_request.max_score = True  # Needed to normalize the scores.
    print(path)
    if path.is_dir():
        gamefiles = list(path.glob("*.ulx"))
    else:
        gamefiles = [path]

    gamefiles = list(map(str, gamefiles))

    # gamefiles = ["/path/to/game.ulx", "/path/to/another/game.z8"]
    action_space = get_action_space(gamefiles)
    observation_space = get_observation_space(gamefiles)
    env_id = register_games(
        gamefiles,
        request_infos=infos_to_request,
        max_episode_steps=max_episode_steps,
        action_space=action_space,
        observation_space=observation_space,
        asynchronous=False,
    )
    env = gym.make(env_id)  # Create a Gym environment to play the text game.
    return env
