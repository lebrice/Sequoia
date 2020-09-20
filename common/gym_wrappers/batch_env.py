import math
from functools import partial
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from typing import (Any, Callable, Iterable, List, Sequence, Tuple, TypeVar,
                    Union)

import gym
import numpy as np
import torch
from baselines.common.vec_env import CloudpickleWrapper, VecEnv
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")
try:
    from baselines.common.vec_env import VecEnv
    from baselines.common.vec_env.subproc_vec_env import worker, SubprocVecEnv
except ImportError:
    raise RuntimeError(
        "Need to have the `baselines` package from openai installed! "
        "Since we're only using it for the SubprocVecEnv wrapper atm, it might "
        "be simplest to install it by doing: \n"
        "`git clone https://github.com/openai/baselines.git` \n"
        "`pip install -e ./baselines`\n."
        "This way, you also won't need to have a Mujoco license, which would be"
        " required to install `baselines` through pip."
)


class _SubprocVecEnv(SubprocVecEnv):
    """ NOTE: @lebrice I'm extending this just so we're able to use a different
    'worker' function in the future if needed.
    """
    def __init__(self, env_fns, spaces=None, worker: Callable=worker):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


class BatchEnv(_SubprocVecEnv, IterableDataset):
    def __init__(self,
                 env: str = None,
                 env_factory: Callable[[], gym.Env] = None,
                 batch_size: int = 1,
                 ):
        assert (isinstance(env, str) or env_factory), (
            "Must pass either a string env or an env_factory must be set."
        )
        if env_factory:
            self.env_factory = env_factory
        else:
            self.env_factory = partial(gym.make, env)
        self.batch_size = batch_size
        # self.envs_per_process = envs_per_process
        env_fns = [
            self.env_factory for _ in range(self.batch_size)
        ]
        super().__init__(env_fns, worker=custom_worker)

    def __getattr__(self, attr: str, default: Any=None):
        print(f"Trying to get missing attribute {attr} (default={default}).")
        for remote in self.remotes:
            remote.send(('getattr', (attr, default)))
        results = [
            remote.recv() for remote in self.remotes
        ]
        return results




def custom_worker(remote: Connection, parent_remote: Connection, env_fn_wrapper):
    """ NOTE: Unused, just copied it here from the subproc_vec_env.py to look at
    it.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    timeout: int = 1
    timeouts = 0
    death: int = 5
    while True:
        ready_objects: List = wait([remote], timeout=timeout)
        if not ready_objects:
            print(f"Worker Received no command for {timeout} seconds.")
            timeouts += 1
            if timeouts == death:
                break
            continue
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == "getattr":
            # Adding this: When asked to get an attribute, get the attr.
            if isinstance(data, str):
                remote.send(getattr(env, data))
            elif len(data) == 1:
                remote.send(getattr(env, data[0]))
            elif len(data) == 2:
                remote.send(getattr(env, data[0], data[1]))
        elif cmd == "setattr":
            # Adding this: When asked to set an attribute, set the attr.
            assert len(data) == 2
            setattr(env, data[0], data[1])
        else:
            raise NotImplementedError
