import itertools
from dataclasses import fields
from typing import Any, ClassVar, Dict, Optional, Type

import gym
import pytest
from sequoia.common.config import Config
from sequoia.conftest import monsterkong_required
from sequoia.methods import Method
from sequoia.settings import Setting
from sequoia.settings.assumptions.incremental_test import DummyMethod as _DummyMethod
from sequoia.settings.rl.envs import (
    ATARI_PY_INSTALLED,
    MONSTERKONG_INSTALLED,
    MUJOCO_INSTALLED,
    MetaMonsterKongEnv,
)

from ..continual.setting_test import TestContinualRLSetting as ContinualRLSettingTests
from ..continual.setting_test import make_dataset_fixture
from .setting import DiscreteTaskAgnosticRLSetting


class TestDiscreteTaskAgnosticRLSetting(ContinualRLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticRLSetting
    dataset: pytest.fixture = make_dataset_fixture(DiscreteTaskAgnosticRLSetting)

    @pytest.fixture(params=[1, 2])
    def nb_tasks(self, request):
        n = request.param
        return n

    @pytest.fixture()
    def setting_kwargs(self, dataset: str, nb_tasks: int, config: Config):
        """ Fixture used to pass keyword arguments when creating a Setting. """
        return {"dataset": dataset, "nb_tasks": nb_tasks, "config": config}

    @pytest.mark.parametrize(
        "dataset, expected_resulting_name",
        [
            ("monsterkong", "MetaMonsterKong-v1"),
            ("meta_monsterkong", "MetaMonsterKong-v1"),
            ("cartpole", "CartPole-v1"),
        ],
    )
    def test_passing_name_variant_works(
        self, dataset: str, expected_resulting_name: str
    ):
        assert self.Setting(dataset=dataset).dataset == expected_resulting_name

    def validate_results(
        self,
        setting: DiscreteTaskAgnosticRLSetting,
        method: Method,
        results: DiscreteTaskAgnosticRLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        assert len(results.task_results) == setting.nb_tasks
        assert [
            sum(task_result.metrics) == task_result.average_metrics
            for task_result in results.task_results
        ]
        assert (
            sum(task_result.average_metrics for task_result in results.task_results)
            == results.average_metrics
        )

    @pytest.mark.parametrize("give_nb_tasks", [True, False])
    @pytest.mark.parametrize("give_train_max_steps", [True, False])
    @pytest.mark.parametrize(
        "give_train_task_schedule, ids_instead_of_steps",
        [(True, False), (True, True), (False, False)],
    )
    @pytest.mark.parametrize(
        "nb_tasks, train_max_steps, train_task_schedule",
        [
            (1, 10_000, {0: {"gravity": 5.0}, 10_000: {"gravity": 10}}),
            (
                4,
                100_000,
                {
                    0: {"gravity": 5.0},
                    25_000: {"gravity": 10},
                    50_000: {"gravity": 10},
                    75_000: {"gravity": 10},
                    100_000: {"gravity": 20},
                },
            ),
        ],
    )
    def test_fields_are_consistent(
        self,
        nb_tasks: Optional[int],
        train_max_steps: Optional[int],
        train_task_schedule: Optional[Dict[str, Any]],
        give_nb_tasks: bool,
        give_train_max_steps: bool,
        give_train_task_schedule: bool,
        ids_instead_of_steps: bool,
    ):

        # give_nb_tasks = True
        # give_max_steps = True
        # give_task_schedule = True
        defaults = {f.name: f.default for f in fields(self.Setting)}
        default_max_train_steps = defaults["train_max_steps"]
        default_nb_tasks = defaults["nb_tasks"]
        # TODO: Same test for test_max_steps?
        full_kwargs = dict(
            nb_tasks=nb_tasks,
            train_max_steps=train_max_steps,
            train_task_schedule=train_task_schedule,
        )
        # for give_nb_task, give_max_steps, give_task_schedule in itertools.product(*[[True, False] for _ in range(3)]):
        kwargs = full_kwargs.copy()
        if not give_nb_tasks:
            kwargs.pop("nb_tasks")
        if not give_train_max_steps:
            kwargs.pop("train_max_steps")
        if not give_train_task_schedule:
            kwargs.pop("train_task_schedule")
        elif ids_instead_of_steps:
            kwargs["train_task_schedule"] = {
                i: task for i, (step, task) in enumerate(train_task_schedule.items())
            }

        setting = self.Setting(**kwargs)
        assert (
            setting.nb_tasks == nb_tasks
            if give_nb_tasks
            else len(train_task_schedule)
            if give_train_task_schedule
            else default_nb_tasks
        )
        assert (
            setting.train_max_steps == train_max_steps
            if give_train_max_steps
            else max(train_task_schedule)
            if give_train_task_schedule
            else default_max_train_steps
        )
        assert list(setting.train_task_schedule.keys()) == [
            i * (setting.train_max_steps / setting.nb_tasks)
            for i in range(0, setting.nb_tasks + 1)
        ]
        assert list(setting.val_task_schedule.keys()) == [
            i * (setting.train_max_steps / setting.nb_tasks)
            for i in range(0, setting.nb_tasks + 1)
        ]
        assert list(setting.test_task_schedule.keys()) == [
            i * (setting.test_max_steps / setting.nb_tasks)
            for i in range(0, setting.nb_tasks + 1)
        ]

        # When giving only the number of tasks:


from typing import Any, Dict, Optional


def test_fit_and_on_task_switch_calls():
    setting = DiscreteTaskAgnosticRLSetting(
        dataset="CartPole-v0",
        # nb_tasks=5,
        # steps_per_task=100,
        train_max_steps=500,
        test_max_steps=500,
        # test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = _DummyMethod()
    _ = setting.apply(method)
    # == 30 task switches in total.
    assert method.n_task_switches == 0
    assert method.n_fit_calls == 1
    assert not method.received_task_ids
    assert not method.received_while_training


@monsterkong_required
@pytest.mark.parametrize(
    "dataset, expected_env_type",
    [
        ("MetaMonsterKong-v0", MetaMonsterKongEnv),
        ("monsterkong", MetaMonsterKongEnv),
        ("PixelMetaMonsterKong-v0", MetaMonsterKongEnv),
        ("monster_kong", MetaMonsterKongEnv),
        ("monster_kong", MetaMonsterKongEnv),
        # ("halfcheetah", ContinualHalfCheetahEnv),
        # ("HalfCheetah-v2", ContinualHalfCheetahV2Env),
        # ("HalfCheetah-v3", ContinualHalfCheetahV3Env),
        # ("ContinualHalfCheetah-v2", ContinualHalfCheetahV2Env),
        # ("ContinualHalfCheetah-v3", ContinualHalfCheetahV3Env),
        # ("ContinualHopper-v2", ContinualHopperEnv),
        # ("hopper", ContinualHopperEnv),
        # ("Hopper-v2", ContinualHopperEnv),
        # ("walker2d", ContinualWalker2dV3Env),
        # ("Walker2d-v2", ContinualWalker2dV2Env),
        # ("Walker2d-v3", ContinualWalker2dV3Env),
        # ("ContinualWalker2d-v2", ContinualWalker2dV2Env),
        # ("ContinualWalker2d-v3", ContinualWalker2dV3Env),
    ],
)
def test_monsterkong_env_name_maps_to_continual_variant(
    dataset: str, expected_env_type: Type[gym.Env]
):
    setting = DiscreteTaskAgnosticRLSetting(
        dataset=dataset, train_max_steps=10_000, test_max_steps=10_000
    )
    train_env = setting.train_dataloader()
    assert isinstance(train_env.unwrapped, expected_env_type)
