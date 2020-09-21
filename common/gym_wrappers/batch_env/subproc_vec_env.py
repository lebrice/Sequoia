"""Modification of subproc_vec_env.py from the openai baselines package used
to customize the 'worker' function. 

Raises:
    RuntimeError: [description]
"""
import multiprocessing as mp
from multiprocessing import Process
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

import numpy as np

from utils import n_consecutive
from utils.logging_utils import get_logger

from .worker import custom_worker, Commands

logger = get_logger(__file__)
_missing = object()

T = TypeVar("T")

try:
    from baselines.common.vec_env import CloudpickleWrapper, VecEnv
    from baselines.common.vec_env.vec_env import clear_mpi_env_vars
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
except ImportError as e:
    raise RuntimeError(
        "Need to have the `baselines` package from openai installed! "
        "Since we're only using it for the SubprocVecEnv wrapper atm, it might "
        "be simplest to install it by doing: \n"
        "`pip install git+https://github.com/openai/baselines.git` \n"
        "This way, you also won't need to have a Mujoco license, which would "
        "be required when installing through pip. Also note, it requires "
        "tensorflow to be installed, for some reason, but you should be able "
        "to just remove it after."
    ) from e

class _SubprocVecEnv(SubprocVecEnv):
    """ NOTE: @lebrice I'm extending this just so we're able to use a different
    'worker' function in the future if needed.
    """
    def __init__(self,
                 env_fns,
                 spaces = None,
                 context: str = "spawn",
                 in_series: int = 1,
                 worker: Callable = custom_worker):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        
        self.ps: List[Process] = []
        for worker_index, (work_remote, remote, env_fn) in enumerate(zip(
                    self.work_remotes, self.remotes, env_fns
                )):
            process = ctx.Process(
                target=worker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                    worker_index
                ),
                # if the main process crashes, we should not cause things to hang
                daemon=True,
            )
            self.ps.append(process)
        for process in self.ps:
            with clear_mpi_env_vars():
                process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send((Commands.get_spaces_spec, None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x

        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

        self.remotes[0].send((Commands.get_attr, "reward_range", None))
        self.reward_range = self.remotes[0].recv().x[0]

    def __getattr__(self, attr: str, default: Any=_missing) -> List:
        logger.debug(f"Trying to get missing attribute '{attr}'.")
        # TODO: This is causing problems atm.
        if default is not _missing:
            try:
                return self.get_attr_from_envs(attr)
            except AttributeError:
                return default
        else:
            return self.get_attr_from_envs(attr)

    def get_attr_from_envs(self, attr: str) -> List:
        for remote in self.remotes:
            remote.send((Commands.get_attr, attr))
        results = []
        for i, remote in enumerate(self.remotes):
            remote_results: Sequence = remote.recv().x
            logger.debug(f"Results from remote #{i}: {remote_results}")
            for result in remote_results:
                if isinstance(result, AttributeError):
                    raise result
                results.append(result)
        logger.debug(f"Results: {results}")
        return results

    def set_attr_on_envs(self, attr: str, value: Any) -> None:
        logger.debug(f"Will try to set attribute '{attr}' to a value of {value} in all environments.")
        for remote in self.remotes:
            remote.send((Commands.set_attr, CloudpickleWrapper((attr, value))))
        for remote in self.remotes:
            # We expect to return a 'None', even though it isn't strictly
            # necessary, just so we know everything went well.
            remote_results: List = remote.recv()
            for result in remote_results:
                if result is not None:
                    raise RuntimeError(
                        f"Something went wrong when trying to set attribute "
                        f"{attr}: {result}"
                    )

    def set_attr_on_each_env(self, attr: str, values: Sequence) -> None:
        logger.debug(f"Will try to set attribute '{attr}' with the corresponding values for each env.")
        # Just in case we're given a generator or iterable.
        values = list(values)
        if len(values) != self.num_envs:
            raise RuntimeError(
                f"You need to pass a value for each of the {self.num_envs} "
                f"environments. (received {values})"
            )

        # Make a list of the values for each remote.
        values_per_remote: List[Tuple] = list(n_consecutive(values, self.in_series))

        for remote, values_for_remote in zip(self.remotes, values_per_remote):
            args = CloudpickleWrapper((attr, values_for_remote))
            remote.send((Commands.set_attr_on_each, args))

        for remote in self.remotes:
            # We expect to return a 'None', even though it isn't strictly
            # necessary, just so we know everything went well.
            remote_results: List = remote.recv()
            for result in remote_results:
                if result is not None:
                    raise RuntimeError(f"Something went wrong when trying to set attribute {attr}: {result}")
