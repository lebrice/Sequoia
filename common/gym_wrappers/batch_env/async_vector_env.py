import multiprocessing as mp
import platform
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import gym
from gym.vector import AsyncVectorEnv as AsyncVectorEnv_

from utils.logging_utils import get_logger

from .worker import CloudpickleWrapper, Commands, custom_worker

logger = get_logger(__file__)
T = TypeVar("T")


class AsyncVectorEnv(AsyncVectorEnv_):
    
    def __init__(self,
                 env_fns: Sequence[Callable[[], gym.Env]],
                 context=None,
                 worker=None,
                 **kwargs):
        if context is None:
            system: str = platform.system()
            if system == "Linux":
                # TODO: Debugging an error from the pyglet package when using 'fork'.
                # python3.7/site-packages/pyglet/gl/xlib.py", line 218, in __init__
                # raise gl.ContextException('Could not create GL context')
                # context = "fork"
                # context = "spawn"
                # NOTE: For now 'forkserver`, seems to have resolved the bug
                # above for now:
                context = "forkserver"
            else:
                logger.warning(RuntimeWarning(
                    f"Using the 'spawn' multiprocessing context since we're on "
                    f"a non-linux {system} system. This means creating new "
                    f"worker processes will probably be quite a bit slower. "
                ))
                context = "spawn"

        # TODO: @lebrice If we want to be able to add back the cool things we
        # had before, like remotely modifying the envs' attributes, only
        # resetting a portion of them, etc, we'll have to take a look at the
        # worker_ function, copy it into `worker.py`, modify it, and then change
        # the value of `worker` here.

        super().__init__(
            env_fns=env_fns,
            context=context,
            worker=worker,
            **kwargs
        )

    def random_actions(self) -> Tuple:
        return self.action_space.sample()
