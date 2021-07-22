from __future__ import annotations

import os
from glob import glob

import gym

import textworld.gym
from textworld import EnvInfos, Game
from textworld.gym.envs import TextworldGymEnv
from textworld.gym.spaces import Char, Word
from textworld.text_utils import extract_vocab, extract_vocab_from_gamefiles


def get_action_space(gamefiles: str | list[str]) -> Word:
    # files = [gamefiles] if isinstance(gamefiles, str) else gamefiles
    # file_paths: list[Path] = list(map(Path, files))
    # for game_file in file_paths:
    #     jsonfile = game_file.with_suffix(".json")
    #     assert jsonfile.exists()
    #     game = Game.load(jsonfile)
    # assert False, game.kb.inform7_commands
    vocab = extract_vocab_from_gamefiles(gamefiles)
    vocab = sorted(vocab)  # Sorting the vocabulary, optional.
    max_length = 8  # TODO: How do we get this max length?
    return Word(max_length=max_length, vocab=vocab)


def get_observation_space(gamefiles: str | list[str]) -> Word:
    # raise NotImplementedError("TODO")
    # TODO: For now this will be the same as the action space.
    return get_action_space(gamefiles)

    # files = [gamefiles] if isinstance(gamefiles, str) else gamefiles
    # file_paths: list[Path] = list(map(Path, files))
    # for game_file in file_paths:
    #     jsonfile = game_file.with_suffix(".json")
    #     assert jsonfile.exists()
    #     game = Game.load(jsonfile)

    # vocab = extract_vocab_from_gamefiles(gamefiles)
    # vocab = sorted(vocab)  # Sorting the vocabulary, optional.
    # return Word(max_length=8, vocab=vocab)


class TextWorldConvenienceWrapper(gym.Wrapper):
    """ WIP: Trying to make a convenience wrapper that just tokenizes everything and has
    sensible observation & action spaces.
    """

    def __init__(self, env: TextworldGymEnv):
        super().__init__(env=env)
        self.env: TextworldGymEnv
        self.observation_space: Word = self.env.observation_space or get_observation_space(
            self.env.gamefiles
        )
        self.action_space: Word = self.env.action_space or get_action_space(
            self.env.gamefiles
        )

        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.env.request_infos = EnvInfos(
            description=True,
            inventory=True,
            admissible_commands=True,
            won=True,
            lost=True,
            max_score=True,
        )
        # assert False, self.action_to_str(self.action_space.sample())

    def action_to_str(self, action: np.ndarray) -> str:
        return " ".join(np.array(self.action_space.vocab)[action])

    def reset(self):
        obs, infos = self.env.reset()
        self._infos = infos
        return obs, infos
        # obs, infos = self.env.reset()
        # IDEA: How about we change the action space dynamically over the course of the
        # game?
        # assert False, infos
        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        assert False, input_tensor
        return input_tensor
        # return obs

    def step(self, action: np.ndarray | str):
        if isinstance(action, np.ndarray):
            action = self.action_to_str(action)
        obs, rewards, done, infos = self.env.step(action)

        # Build agent's observation: feedback + look + inventory.
        input_ = "{}\n{}\n{}".format(
            obs, self._infos["description"], self._infos["inventory"]
        )
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        assert False, infos

        commands_tensor = self._process(infos["admissible_commands"])

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]

    def _tokenize(self, text: str) -> list[int]:
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, : len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0)  # Batch x Seq => Seq x Batch
        return padded_tensor

