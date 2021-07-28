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
from textworld.gym.spaces import Char, Word
from textworld.text_utils import extract_vocab, extract_vocab_from_gamefiles
from textworld.gym.envs import TextworldGymEnv, TextworldBatchGymEnv
from typing import Mapping, Any
import numpy as np


class RandomAgent(Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return self.rng.choice(infos["admissible_commands"])


class RandomTextworldMethod(Method, target_setting=RLSetting):
    def __init__(self):
        super().__init__()
        self.agent: Agent

    def configure(self, setting: RLSetting):
        self.agent = RandomAgent()

    def fit(self, train_env: TextworldGymEnv, valid_env: TextworldGymEnv):
        self.train(train_env)
        self.play(valid_env)

    def play(
        self, env: TextworldGymEnv, nb_episodes=10, verbose=True
    ):
        agent = self.agent
        max_episode_steps = env.max_episode_steps #100
        # env = TextWorldConvenienceWrapper(env)


        # Collect some statistics: nb_steps, final reward.
        avg_moves, avg_scores, avg_norm_scores = [], [], []
        for no_episode in range(nb_episodes):
            obs, infos = env.reset()  # Start new episode.

            score = 0
            done = False
            nb_moves = 0
            while not done:
                command = agent.act(obs, score, done, infos)
                obs, score, done, infos = env.step(command)
                nb_moves += 1

            agent.act(obs, score, done, infos)  # Let the agent know the game is done.

            if verbose:
                print(".", end="")
            avg_moves.append(nb_moves)
            avg_scores.append(score)
            avg_norm_scores.append(score / infos["max_score"])

        env.close()
        msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
        if verbose:
            if os.path.isdir(path):
                print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
            else:
                print(
                    msg.format(
                        np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]
                    )
                )

