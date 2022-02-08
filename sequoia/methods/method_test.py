from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, Type, TypeVar

import pytest

from sequoia.common.config import Config
from sequoia.settings import RLSetting, Setting, SLSetting
from sequoia.settings.base import Method
from sequoia.settings.sl.continual.setting import random_subset
from sequoia.conftest import session_config, config


def key_fn(setting_class: Type[Setting]):
    # order tests in terms of their 'depth' in the tree, and break ties arbitrarily
    # based on the name.
    return (len(setting_class.parents()), setting_class.__name__)


def make_setting_type_fixture(method_type: Type[Method]) -> pytest.fixture:
    """Create a parametrized fixture that will go through all the applicable settings
    for a given method.
    """

    def setting_type(self, request):
        setting_type = request.param
        return setting_type

    setting_types = set(method_type.get_applicable_settings())
    settings_to_remove = set([Setting, SLSetting, RLSetting])
    # NOTE: Need to make a deterministic ordering of settings, otherwise we can't
    # parallelize tests with pytest-xdist
    setting_types = sorted(list(setting_types - settings_to_remove), key=key_fn)
    return pytest.fixture(
        params=setting_types,
        scope="module",
    )(setting_type)


MethodType = TypeVar("MethodType", bound=Method)


class MethodTests(ABC):
    """Base class that can be extended to generate tests for a method.

    The main test of interest is `test_debug`.
    """

    Method: ClassVar[Type[MethodType]]
    setting_type: pytest.fixture
    # Kwargs to pass when contructing the Settings.
    setting_kwargs: ClassVar[Dict] = {}
    method_debug_kwargs: ClassVar[Dict] = {}

    def __init_subclass__(cls, method: Type[MethodType] = None):
        """Dynamically generates a `setting_type` fixture on the subclass, which will
        be parametrized by the settings that the Method is applicable to.
        """
        super().__init_subclass__()
        if not method and not hasattr(cls, "Method"):
            raise RuntimeError(
                "Need to either pass `method` when subclassing or set "
                "a 'Method' class attribute."
            )
        cls.Method = cls.Method or method
        cls.setting_type: pytest.fixture = make_setting_type_fixture(cls.Method)

    @classmethod
    @abstractmethod
    @pytest.fixture
    def method(cls, config: Config) -> MethodType:
        """Fixture that returns the Method instance to use when testing/debugging.

        Needs to be implemented when creating a new test class (to generate tests for a
        new method).
        """
        return cls.Method()

    @abstractmethod
    def validate_results(
        self,
        setting: Setting,
        method: MethodType,
        results: Setting.Results,
    ) -> None:
        assert results
        assert results.objective
        assert results.objective is not None
        print(results.summary())

    # NOTE: Need to re-define these here, just so external packages, which maybe aren't
    # in the "scope" of `sequoia/conftest.py` can also use them:
    # Dropping the `self` argument by making those static methods on the class.
    session_config: pytest.fixture = staticmethod(session_config)
    config: pytest.fixture = staticmethod(config)

    @pytest.fixture(scope="module")
    def setting(self, setting_type: Type[Setting], session_config: Config):
        # TODO: Fix this test setup, nb_tasks should be something low like 2, and
        # perhaps use max_episode_steps to limit episode length
        if issubclass(setting_type, SLSetting):
            setting_kwargs = dict(
                nb_tasks=5,
                config=session_config,
            )
            setting_kwargs.setdefault("monitor_training_performance", True)
            # TODO: Do we also want to parameterize the dataset? or is it too much?
            setting_kwargs.update(self.setting_kwargs)
            setting = setting_type(
                **setting_kwargs,
            )
            setting.config = session_config
            setting.batch_size = 10
            setting.prepare_data()
            setting.setup()
            nb_tasks = 5
            samples_per_task = 50
            # Testing this out: Shortening the train datasets:
            setting.train_datasets = [
                random_subset(task_dataset, samples_per_task)
                for task_dataset in setting.train_datasets
            ]
            setting.val_datasets = [
                random_subset(task_dataset, samples_per_task)
                for task_dataset in setting.val_datasets
            ]
            setting.test_datasets = [
                random_subset(task_dataset, samples_per_task)
                for task_dataset in setting.test_datasets
            ]
            assert len(setting.train_datasets) == nb_tasks
            assert len(setting.val_datasets) == nb_tasks
            assert len(setting.test_datasets) == nb_tasks
            assert all(len(dataset) == samples_per_task for dataset in setting.train_datasets)
            assert all(len(dataset) == samples_per_task for dataset in setting.val_datasets)
            assert all(len(dataset) == samples_per_task for dataset in setting.test_datasets)

            # Assert that calling setup doesn't overwrite the datasets.
            setting.setup()
            assert len(setting.train_datasets) == nb_tasks
            assert len(setting.val_datasets) == nb_tasks
            assert len(setting.test_datasets) == nb_tasks
            assert all(len(dataset) == samples_per_task for dataset in setting.train_datasets)
            assert all(len(dataset) == samples_per_task for dataset in setting.val_datasets)
            assert all(len(dataset) == samples_per_task for dataset in setting.test_datasets)
        else:
            # RL setting:
            setting_kwargs = dict(
                nb_tasks=2,
                train_max_steps=1_000,
                test_max_steps=1_000,
                # train_steps_per_task=2_000,
                # test_steps_per_task=1_000,
                config=session_config,
            )
            # TODO: Do we also want to parameterize the dataset? or is it too much?
            setting_kwargs.update(self.setting_kwargs)
            setting = setting_type(
                **setting_kwargs,
            )

        yield setting

    def test_debug(self, method: MethodType, setting: Setting, config: Config):
        """Apply the Method onto a setting, and validate the results."""
        results: Setting.Results = setting.apply(method, config=config)
        self.validate_results(setting=setting, method=method, results=results)


@dataclass
class NewSetting(Setting):
    pass


@dataclass
class NewMethod(Method, target_setting=NewSetting):
    def fit(self, train_env, valid_env):
        pass

    def get_actions(self, observations, action_space):
        return action_space.sample()


def test_passing_arg_to_class_constructor_works():
    assert NewMethod.target_setting is NewSetting
    assert NewMethod().target_setting is NewSetting


@pytest.mark.xfail(reason="Not sure this is necessary.")
def test_cant_change_target_setting():
    with pytest.raises(AttributeError):
        NewMethod.target_setting = NewSetting
    with pytest.raises(AttributeError):
        NewMethod().target_setting = NewSetting


def test_target_setting_is_inherited():
    @dataclass
    class NewMethod2(NewMethod):
        pass

    assert NewMethod2.target_setting is NewSetting


@dataclass
class SettingA(Setting):
    pass


@dataclass
class SettingA1(SettingA):
    pass


@dataclass
class SettingA2(SettingA):
    pass


@dataclass
class SettingB(Setting):
    pass


class MethodA(Method, target_setting=SettingA):
    def fit(self, train_env, valid_env):
        pass

    def get_actions(self, observations, action_space):
        return action_space.sample()


class MethodB(Method, target_setting=SettingB):
    def fit(self, train_env, valid_env):
        pass

    def get_actions(self, observations, action_space):
        return action_space.sample()


class CoolGeneralMethod(Method, target_setting=Setting):
    def fit(self, train_env, valid_env):
        pass

    def get_actions(self, observations, action_space):
        return action_space.sample()


def test_method_is_applicable_to_setting():
    """Test the mechanism for determining if a method is applicable for a given
    setting.

    Uses the mock hierarchy created above:
    - Setting
        - SettingA
            - SettingA1
            - SettingA2
        - SettingB

    - Method
        - MethodA (target_setting: SettingA)
        - MethodB (target_setting: SettingA)

    TODO: if we ever end up registering the method classes when declaring them,
    then we will need to check that this dummy test hierarchy doesn't actually
    show up in the real setting options.
    """
    # A Method designed for `SettingA` ISN'T applicable on the root node
    # `Setting`:
    assert not MethodA.is_applicable(Setting)

    # A Method designed for `SettingA` IS applicable on the target node, and all
    # nodes below it in the tree:
    assert MethodA.is_applicable(SettingA)
    assert MethodA.is_applicable(SettingA1)
    assert MethodA.is_applicable(SettingA2)
    # A Method designed for `SettingA` ISN'T applicable on some other branch in
    # the tree:
    assert not MethodA.is_applicable(SettingB)

    # Same for Method designed for `SettingB`
    assert MethodB.is_applicable(SettingB)
    assert not MethodB.is_applicable(Setting)
    assert not MethodB.is_applicable(SettingA)
    assert not MethodB.is_applicable(SettingA1)
    assert not MethodB.is_applicable(SettingA2)


def test_is_applicable_also_works_on_instances():
    assert MethodA().is_applicable(SettingA)
    assert MethodA.is_applicable(SettingA())
    assert MethodA().is_applicable(SettingA())

    assert not MethodA().is_applicable(SettingB)
    assert not MethodA.is_applicable(SettingB())
    assert not MethodA().is_applicable(SettingB())
