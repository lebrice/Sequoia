import argparse
import os
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Tuple, Type, Union

import pytest

import sys; sys.path += [os.path.abspath('..'), os.path.abspath('.')]
from common.config import Config, TrainerConfig
from methods.method import Method
from settings import Setting
from simple_parsing import Serializable

parametrize = pytest.mark.parametrize

xfail = pytest.mark.xfail


def xfail_param(*args, reason: str):
    return pytest.param(*args, marks=pytest.mark.xfail(reason=reason))

def skip_param(*args, reason: str):
    return pytest.param(*args, marks=pytest.mark.skip(reason=reason))


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

@pytest.fixture()
def config(tmp_path: Path):
    return Config(debug=True, data_dir="data", log_dir_root=tmp_path)


def get_all_dataset_names(method_class: Type[Method]) -> List[str]:
    dataset_names: Iterable[List[str]] = map(
        lambda s: list(s.available_datasets),
        method_class.get_all_applicable_settings()
    )
    return list(set(sum(dataset_names, []))) 


def get_dataset_params(method_type: Type[Method],
                       supported_datasets: List[str],
                       skip_unsuported: bool=True) -> List[str]:
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


def make_fixtures(method_class: Type[Method], argv: Union[str, List[str]] = None) -> Tuple[pytest.fixture, pytest.fixture]:
    method = make_method_fixture(method_class, argv=argv)
    setting = make_setting_fixtures(method_class, argv=argv)
    return method, setting


def make_method_fixture(method_class: Type[Method],
                        argv: Union[str, List[str]] = None,
                        fixture_name: str = "method") -> pytest.fixture:
    if not argv:
        # create the default args if not given.
        argv = f"""
            --knn_samples 0
        """
    @pytest.fixture(name=fixture_name)
    def method_fixture(trainer_config: TrainerConfig, config: Config):
        method: Method
        if argv:
            method = method_class.from_args(argv)
        else:
            method = method_class()
        method.trainer_options = trainer_config
        method.config = config
        return method
    return method_fixture



def make_method_factory_fixture(method_class: Type[Method]) -> pytest.fixture:
    """Creates a fixture that will give a 'constructor' for the given method type.
    
    This constructor can be passed some HParams and will return an instance of
    the given method class.
    """
    @pytest.fixture()
    def method_factory(trainer_options: TrainerConfig, config: Config):
        def _method(hparams: Union[str, List[str], Serializable] = None) -> Method:
            if isinstance(hparams, (str, list)):
                # should we parse the hparams if given a string?
                hparams_type = type(method_class.hparams)
                hparams = hparams_type.from_args(hparams)
                assert False, hparams_type

            return method_class(
                hparams=hparams,
                trainer_options=trainer_options,
                config=config,
            )
        return _method
    return method_factory


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False)


slow = pytest.mark.skipif(
    "--slow" not in sys.argv,
    reason="This test is slow so we only run it when necessary."
)


## TODO: Figure out how to use this properly!
# def pytest_generate_tests(metafunc):
#     # This is called for every test. Only get/set command line arguments
#     # if the argument is specified in the list of test "fixturenames".
#     option_value = metafunc.config.option.name
#     if 'name' in metafunc.fixturenames and option_value is not None:
#         metafunc.parametrize("name", [option_value])
# def pytest_generate_tests(metafunc):
#     if "dataset" in metafunc.fixturenames:
#         datasets: List[str] = metafunc.config.getoption("datasets")
#         # metafunc.
#         metafunc.parametrize("dataset", datasets, indirect=True)


## TODO: Was trying to include the dataset name into the name of each individual
## run in the 'test runs view' instead of the index of the fixture parameter.
# from pytest import Item
# def pytest_itemcollected(item: Item):
#     """ change test name, using fixture names """
#     if item._fixtureinfo.argnames:
#         item._nodeid = ', '.join(item._fixtureinfo.argnames)
