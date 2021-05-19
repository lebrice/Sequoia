import argparse
import json
import logging
import os
import sys

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import gym
import pytest
from simple_parsing import Serializable

from sequoia.common.config import Config, TrainerConfig
from sequoia.settings import Method, Setting


logger = logging.getLogger(__file__)

parametrize = pytest.mark.parametrize

xfail = pytest.mark.xfail


def xfail_param(*args, reason: str):
    return pytest.param(*args, marks=pytest.mark.xfail(reason=reason))


def skip_param(*args, reason: str):
    return pytest.param(*args, marks=pytest.mark.skip(reason=reason))


def skipif_param(condition, *args, reason: str):
    return pytest.param(*args, marks=pytest.mark.skipif(condition, reason=reason))


@pytest.fixture()
def trainer_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("log_dir")
    return TrainerConfig(
        fast_dev_run=True,
        # TODO: What if we don't have a GPU when testing?
        # TODO: Parametrize with the distributed backend, skip param if no GPU?
        distributed_backend="dp",
        default_root_dir=tmp_path,
    )


@pytest.fixture(scope="session")
def config():
    # TODO: Set the results dir somehow with the value of this `tmp_path` fixture.
    data_dir = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("DATA_DIR", "data")))
    return Config(debug=True, data_dir=data_dir, seed=123,)


def id_fn(params: Any) -> str:
    """Creates a 'name' for an execution of a parametrized test.

    Args:
        params (Dict): [description]

    Returns:
        str: [description]
    """
    # if not params:
    #     return "default"
    if isinstance(params, dict):
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    return str(params)


def get_all_dataset_names(method_class: Type[Method] = None) -> List[str]:
    # When not given a method class, use the Method class (gives ALL the
    # possible datasets).
    method_class = method_class or Method

    dataset_names: Iterable[List[str]] = map(
        lambda s: list(s.available_datasets), method_class.get_applicable_settings()
    )
    return list(set(sum(dataset_names, [])))


def get_dataset_params(
    method_type: Type[Method],
    supported_datasets: List[str],
    skip_unsuported: bool = True,
) -> List[str]:
    all_datasets = get_all_dataset_names(method_type)
    dataset_params = []
    for dataset in all_datasets:
        if dataset in supported_datasets:
            dataset_params.append(dataset)
        elif skip_unsuported:
            dataset_params.append(skip_param(dataset, reason="Not supported yet"))
        else:
            dataset_params.append(xfail_param(dataset, reason="Not supported yet"))
    return dataset_params


test_datasets_option_name: str = "datasets"


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False)
    parser.addoption(
        f"--{test_datasets_option_name}", action="store", nargs="*", default=[]
    )


slow = pytest.mark.skipif(
    "--slow" not in sys.argv,
    reason="This test is slow so we only run it when necessary.",
)


def slow_param(*args):
    return pytest.param(*args, marks=slow)


def find_class_under_test(
    module, function, name: str = "method", global_var_name: str = None
) -> Optional[Type]:
    cls: Optional[Type] = None
    module_name: str = module.__name__
    function_name: str = function.__name__
    type_hints = get_type_hints(function)
    global_var_name = global_var_name or name.capitalize()
    for k in [name, f"{name}_class", f"{name}_type"]:
        cls = type_hints.get(k)
        if cls:
            logger.debug(
                f"function {function_name} has annotation of type "
                f"{cls} for argument {k}."
            )
            break
    if cls is None:
        # Try to get the class to test from a global variable on the module.
        cls = getattr(module, global_var_name, None)
        logger.debug(
            f"Test module {module_name} has a '{global_var_name}' gloval variable of type {cls}"
        )
    return cls


def parametrize_test_datasets(metafunc):
    # We want to get these from inspecting the test function:
    # The datasets to test on.
    test_datasets: List[str] = []
    default_test_datasets = ["mnist", "cifar10"]
    func_param_name = "test_dataset"
    global_var_names = ["test_datasets", "supported_datasets"]

    if func_param_name not in metafunc.fixturenames:
        return

    module = metafunc.module
    function = metafunc.function

    module_name: str = module.__name__
    function_name: str = function.__name__

    # Get the test datasets from the command-line option.
    datasets_from_command_line = metafunc.config.getoption(test_datasets_option_name)

    if "ALL" in datasets_from_command_line:
        method_class: Optional[Type[Method]] = find_class_under_test(
            module, function, name="method",
        )
        test_datasets = get_all_dataset_names(method_class)
    elif "NONE" in datasets_from_command_line:
        test_datasets = [skip_param("?", reason="Set to skip, with command line arg.")]
    elif datasets_from_command_line:
        assert isinstance(datasets_from_command_line, list) and all(
            isinstance(v, str) for v in datasets_from_command_line
        )
        # If any datasets were set, use them.
        test_datasets = datasets_from_command_line
    else:
        # The default datasets to try are the ones specified at the global
        # variable with name {module_test_datasets_name} in the module.
        for global_var_name in global_var_names:
            test_datasets = getattr(module, global_var_name, None)
            if test_datasets is not None:
                break
        else:
            logger.warning(
                RuntimeWarning(
                    f"Test module {module_name} didn't specify a test_datasets "
                    f"global variable, defaulting to {default_test_datasets}"
                )
            )
            test_datasets = default_test_datasets

    logger.info(
        f"Parametrizing the '{func_param_name}' param of test "
        f"{module_name} :: {function_name} with {test_datasets}."
    )
    metafunc.parametrize(func_param_name, test_datasets)


def pytest_generate_tests(metafunc):
    """ Automatically Parametrize the tests.
    TODO: Having some fun parametrizing tests automatically, but should check
    that it's worth it, because otherwise it might make things too confusing. 
    """
    parametrize_test_datasets(metafunc)


class DummyEnvironment(gym.Env):
    """ Dummy environment for testing.
    
    The reward is how close to the target value the state (a counter) is. The
    actions are:
    0:  keep the counter the same.
    1:  Increment the counter.
    2:  Decrement the counter.
    """

    def __init__(self, start: int = 0, target: int = 5, max_value: int = None):
        self.i = start
        self.start = start
        max_value = max_value if max_value is not None else target * 2
        assert 0 <= target <= max_value
        self.max_value = max_value
        self.reward_range = (0, max_value)
        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Discrete(n=max_value)

        self.target = target
        self.reward_range = (0, max(target, max_value - target))

        self.done: bool = False
        self._reset: bool = False

    def step(self, action: int):
        # The action modifies the state, producing a new state, and you get the
        # reward associated with that transition.
        if not self._reset:
            raise RuntimeError("Need to reset before you can step.")
        if action == 1:
            self.i += 1
        elif action == 2:
            self.i -= 1
        self.i %= self.max_value
        done = self.i == self.target
        reward = abs(self.i - self.target)
        # print(self.i, reward, done, action)
        return self.i, reward, done, {}

    def reset(self):
        self._reset = True
        self.i = self.start
        return self.i

    def seed(self, seed: Optional[int]) -> List[int]:
        seeds = []
        seeds.append(self.observation_space.seed(seed))
        seeds.append(self.action_space.seed(seed))
        return seeds


from sequoia.settings.active.envs import (
    METAWORLD_INSTALLED,
    MONSTERKONG_INSTALLED,
    MTENV_INSTALLED,
    ATARI_PY_INSTALLED,
    MUJOCO_INSTALLED,
)

monsterkong_required = pytest.mark.skipif(
    not MONSTERKONG_INSTALLED, reason="monsterkong is required for this test."
)


def param_requires_monsterkong(*args):
    return skipif_param(
        not MONSTERKONG_INSTALLED,
        *args,
        reason="monsterkong is required for this parameter.",
    )


atari_py_required = pytest.mark.skipif(
    not ATARI_PY_INSTALLED, reason="atari_py is required for this test."
)


def param_requires_atari_py(*args):
    return skipif_param(
        not ATARI_PY_INSTALLED,
        *args,
        reason="atari_py is required for this parameter.",
    )


mtenv_required = pytest.mark.skipif(
    not MTENV_INSTALLED, reason="mtenv is required for this test."
)


def param_requires_mtenv(*args):
    return skipif_param(
        not MTENV_INSTALLED, *args, reason="mtenv is required for this parameter.",
    )


metaworld_required = pytest.mark.skipif(
    not METAWORLD_INSTALLED, reason="metaworld is required for this test."
)


def param_requires_metaworld(*args):
    return skipif_param(
        not METAWORLD_INSTALLED,
        *args,
        reason="metaworld is required for this parameter.",
    )


mujoco_required = pytest.mark.skipif(
    not MUJOCO_INSTALLED, reason="mujoco-py is required for this test."
)


def param_requires_mujoco(*args):
    return skipif_param(
        not MUJOCO_INSTALLED, *args, reason="mujoco-py is required for this parameter.",
    )

