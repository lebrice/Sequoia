"""Modification of subproc_vec_env.py from the openai baselines package used
to customize the 'worker' function. 

Raises:
    RuntimeError: [description]
"""
import multiprocessing as mp
import platform
from inspect import ismethod
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import numpy as np

from utils.logging_utils import get_logger
from utils.utils import n_consecutive

from .worker import Commands, custom_worker

logger = get_logger(__file__)
_missing = object()

T = TypeVar("T")

try:
    from baselines.common.vec_env import CloudpickleWrapper, VecEnv
    from baselines.common.vec_env.vec_env import clear_mpi_env_vars
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, worker as _worker
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

from gym.vector import AsyncVectorEnv

class _SubprocVecEnv(SubprocVecEnv):
    """ NOTE: @lebrice I'm extending this just so we're able to use a different
    'worker' function in the future if needed.

    TODO: OMG I just found `gym.vector.
    """
    def __init__(self,
                 env_fns,
                 spaces = None,
                 context: str = None,
                 in_series: int = 1,
                 worker: Callable = custom_worker):
        """
        envs: list of gym environments to run in subprocesses
        """
        if context is None:
            system: str = platform.system()
            if system == "Linux":
                # TODO: Debugging an error from the pyglet package when using 'fork'.
                # python3.7/site-packages/pyglet/gl/xlib.py", line 218, in __init__
                # raise gl.ContextException('Could not create GL context')
                # context = "fork"
                # context = "spawn"
                # NOTE: Testing out `forkserver`, seems to have resolved the bug
                # above for now:
                context = "forkserver"
            else:
                logger.warning(RuntimeWarning(
                    f"Using the 'spawn' multiprocessing context since we're on "
                    f"a non-linux {system} system. This means creating new "
                    f"worker processes will probably be quite a bit slower. "
                ))
                context = "spawn"

        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)

        # remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.remotes: List[Connection] = []
        self.work_remotes: List[Connection] = []

        self.ps: List[Process] = []
        for worker_index, env_fn in enumerate(env_fns):
            remote, worker_remote = ctx.Pipe()

            self.remotes.append(remote)
            self.work_remotes.append(worker_remote)
            worker_args = [
                worker_remote, remote, CloudpickleWrapper(env_fn)
            ]
            if worker is not _worker:
                # Pass the worker_index to the worker args, in case that might
                # be useful. NOTE: The worker_index arg isn't in the normal
                # `worker` function, so this is only done when using the
                # custom_worker function.
                worker_args.append(worker_index)

            process = ctx.Process(
                target=worker,
                args=worker_args,
                # Kill the worker if the main process crashes:
                daemon=True,
            )
            self.ps.append(process)

        for process in self.ps:
            with clear_mpi_env_vars():
                process.start()
        
        # TODO: Why do we close the work remotes here?
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send((Commands.get_spaces_spec, None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        logger.debug(f"Env spec: {self.spec}")
        
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

        self.remotes[0].send((Commands.get_attr, "reward_range", None))
        self.reward_range: Tuple[float, float] = self.remotes[0].recv().x[0]

    def __getattr__(self, attr: str, default: Any=_missing) -> Union[Any, List[Any], Callable]:
        logger.debug(f"Trying to get missing attribute '{attr}'.")
        # TODO: This is causing problems atm.
        attributes: List = []
        if default is _missing:
            attributes = self.get_attr_from_envs(attr)
        else:
            try:
                attributes = self.get_attr_from_envs(attr)
            except AttributeError:
                return default

        logger.debug(f"Attributes: {attributes}")
        # TODO: Having some fun here, should turn keep this off just in case
        # there's any problem. 
        if False and all(map(ismethod, attributes)):
            logger.warning(RuntimeWarning(
                f"The '{attr}' attribute is a method on all envs, returning a "
                "'batched' method, just for fun's sake."
            ))
            from .batched_method import make_batched_method
            return make_batched_method(attributes)
        return attributes

    def get_attr_from_envs(self, attr: str) -> List[Any]:
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

    def partial_reset(self, reset_mask: Sequence[bool]) -> List[Optional[Any]]:
        values_per_remote: List[Tuple[bool, ...]] = self.split_values(reset_mask)
        # Make a list of the values for each remote.
        self._assert_not_closed()
        for remote, values_for_remote in zip(self.remotes, values_per_remote):
            args = CloudpickleWrapper(values_for_remote)
            remote.send((Commands.partial_reset, args))

        results = []
        for remote in self.remotes:
            # We expect to receive None for the envs that weren't reset, and the
            # reset state for those that were.
            remote_results: List = remote.recv()
            if isinstance(remote_results, CloudpickleWrapper):
                remote_results = remote_results.x
            results.extend(remote_results)
        return zip(*results)

    def split_values(self, values: List[T]) -> List[Tuple[T, ...]]:
        # Make a list of the values for each remote.
        values = list(values) # in case it's a generator or something.
        if len(values) != self.num_envs:
            raise RuntimeError(
                f"You need to pass a value for each of the {self.num_envs} "
                f"environments, only received {len(values)} values."
            )
        values_per_remote = list(n_consecutive(values, self.in_series))
        return values_per_remote

    def seed(self, seeds: Union[int, Iterable[int]]) -> None:
        if isinstance(seeds, int):
            seeds = [seeds] * self.num_envs
        seeds = self.split_values(seeds)
        for remote, seeds_for_remote in zip(self.remotes, seeds):
            remote.send((Commands.seed, seeds_for_remote))
        for remote in self.remotes:
            remote.recv()