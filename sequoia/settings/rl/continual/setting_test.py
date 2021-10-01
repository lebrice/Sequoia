import dataclasses
from dataclasses import asdict, is_dataclass, replace
from functools import partial, singledispatch
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type

import gym
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gym import spaces
from gym.vector.utils import batch_space
from sequoia.common.config import Config
from sequoia.common.spaces import TypedDictSpace
from sequoia.common.spaces.sparse import Sparse
from sequoia.conftest import (
    MUJOCO_INSTALLED,
    mujoco_required,
    param_requires_monsterkong,
    param_requires_mujoco,
)
from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
from sequoia.settings.assumptions.incremental_test import DummyMethod as _DummyMethod
from sequoia.settings.rl.setting_test import DummyMethod
from sequoia.utils.utils import pairwise, take
from sequoia.settings.base.setting_test import SettingTests

from .setting import ContinualRLSetting


@pytest.mark.parametrize(
    "dataset",
    [
        "CartPole-v8",
        "Breakout-v9",
        param_requires_mujoco("Ant-v0"),
        param_requires_monsterkong("MetaMonsterKong-v0"),
    ],
)
def test_passing_unsupported_dataset_raises_error(dataset: Any):
    with pytest.raises((gym.error.Error, NotImplementedError)):
        _ = ContinualRLSetting(dataset=dataset)


@singledispatch
def _equal(a: Any, b: Any) -> bool:
    """Utility function used to check if two thing are equal.

    NOTE: This is only really useful/necessary because `functools.partial` objects can be present
    as attributes on the setting, usually either in the task schedule (or in the
    [train/val/test]_envs for the IncrementalRLSetting subclasses).
    The `functools.partial` class doesn't support equality: two partial objects with the same funcs,
    args and kwargs are still not considered equal for some reason.

    This function has a special handler for `partial` objects, so that they are considered equal if
    and only if their funcs, args and keywords are the same.
    This makes it possible to easily check for equality between settings, which is used for example
    in the tests below.
    """
    if is_dataclass(a):
        return is_dataclass(b) and _equal(asdict(a), asdict(b))
    return a == b


@_equal.register
def _partials_equal(a: partial, b: partial) -> bool:
    # NOTE: Using the recursive call so we can compare nested partials.
    return (
        isinstance(b, partial)
        and _equal(a.func, b.func)
        and _equal(a.args, b.args)
        and _equal(a.keywords, b.keywords)
    )


# NOTE: Need to also register handlers for list and dict, since they might have partials as
# items.
@_equal.register(list)
def _lists_equal(a: List, b: List) -> bool:
    return len(a) == len(b) and all(_equal(v_a, v_b) for v_a, v_b in zip(a, b))


@_equal.register(dict)
def _dicts_equal(a: Dict, b: Dict) -> bool:
    if a.keys() != b.keys():
        return False

    for k in a:
        v_a, v_b = a[k], b[k]
        if not _equal(v_a, v_b):
            print(f"Values differ at key {k}: {v_a}, {v_b}")
            return False
    return True


def all_different_from_next(sequence: Sequence) -> bool:
    """Returns True if each value in the sequence is different from the next."""
    return not any(_equal(v, next_v) for v, next_v in pairwise(sequence))


class TestContinualRLSetting(SettingTests):
    Setting: ClassVar[Type[Setting]] = ContinualRLSetting
    dataset: pytest.fixture

    @pytest.fixture()
    def setting_kwargs(self, dataset: str, config: Config):
        """Fixture used to pass keyword arguments when creating a Setting."""
        return {"dataset": dataset, "config": config}

    def test_passing_supported_dataset(self, setting_kwargs: Dict):
        setting = self.Setting(**setting_kwargs)
        assert setting.train_task_schedule
        assert setting.val_task_schedule
        assert setting.test_task_schedule
        # Passing the dataset created a task schedule.
        assert all(setting.train_task_schedule.values()), "Should have non-empty tasks."
        assert all(setting.val_task_schedule.values()), "Should have non-empty tasks."
        assert all(setting.test_task_schedule.values()), "Should have non-empty tasks."

    @pytest.mark.parametrize("seed", [123, 456])
    def test_task_schedule_is_reproducible(self, dataset: str, seed: Optional[int]):
        setting_a = self.Setting(dataset=dataset, config=Config(seed=seed))
        setting_b = self.Setting(dataset=dataset, config=Config(seed=seed))
        assert setting_a.train_task_schedule == setting_b.train_task_schedule
        assert setting_a.val_task_schedule == setting_b.val_task_schedule
        assert setting_a.test_task_schedule == setting_b.test_task_schedule

    @pytest.mark.xfail(
        reason="Reworking/removing this mechanism, makes things a bit too complicated."
    )
    def test_using_deprecated_fields(self):
        # BUG: It's tough to get this to raise a warning, because it's happening
        # inside the constructor in the dataclasses.py file, so we have to mess with
        # descriptors etc, which isn't great.
        # with pytest.raises(DeprecationWarning):
        #     setting = self.Setting(nb_tasks=5, max_steps=123)
        setting = self.Setting(nb_tasks=5, max_steps=123)
        assert setting.train_max_steps == 123

        with pytest.warns(DeprecationWarning):
            setting.max_steps = 456
        assert setting.train_max_steps == 456

        with pytest.warns(DeprecationWarning):
            setting = self.Setting(nb_tasks=5, test_max_steps=123)
        assert setting.test_max_steps == 123

        with pytest.warns(DeprecationWarning):
            setting.test_steps = 456
        assert setting.test_max_steps == 456

    def test_tasks_are_different(self, setting_kwargs: Dict[str, Any], config: Config):
        """Check that the tasks different from the next."""
        config = setting_kwargs.pop("config", config)
        assert config.seed is not None
        setting = self.Setting(**setting_kwargs, config=config)

        # Check that each task is different from the next.
        assert all_different_from_next(setting.train_task_schedule.values())
        assert all_different_from_next(setting.val_task_schedule.values())
        assert all_different_from_next(setting.test_task_schedule.values())

    def test_settings_attributes_are_the_same_for_given_seed(
        self, setting_kwargs: Dict[str, Any], config: Config
    ):
        """Make sure that the settings' attributes are the same if passed the same seed."""
        # Make sure that there is a random seed set, otherwise use the one present in `config`.
        config: Config = setting_kwargs.pop("config", config)
        assert config.seed is not None
        setting_1 = self.Setting(**setting_kwargs, config=config)

        # Uses the same config and seed, and check that the attributes of the two settings are
        # identical.
        setting_2 = self.Setting(**setting_kwargs, config=config)

        # Check that the settings have the same attributes.
        assert _equal(dataclasses.asdict(setting_1), dataclasses.asdict(setting_2))

        # These next lines are redundant, but just to be clear:
        assert setting_1.train_task_schedule == setting_2.train_task_schedule
        assert setting_1.val_task_schedule == setting_2.val_task_schedule
        assert setting_1.test_task_schedule == setting_2.test_task_schedule

    def test_tasks_are_different_when_seed_is_different(
        self, setting_kwargs: Dict[str, Any], config: Config
    ):
        # Create another setting with a different seed, and check that at least the generated tasks
        # are different.
        config = setting_kwargs.pop("config", config)
        assert config.seed is not None
        setting_1 = self.Setting(**setting_kwargs, config=config)
        assert setting_1.train_task_schedule

        different_seed = config.seed + 123
        setting_3 = self.Setting(**setting_kwargs, config=replace(config, seed=different_seed))

        setting_1_dict = dataclasses.asdict(setting_1)
        setting_3_dict = dataclasses.asdict(setting_3)

        # Remove the seeds, which are obviously different, and then check that the dicts from the
        # two settings are still different.
        assert setting_1_dict["config"].pop("seed") == config.seed
        assert setting_3_dict["config"].pop("seed") == different_seed
        if "LPG-FTW" in setting_1.dataset:
            # NOTE: The rest of the setting's attributes might be identical (they currently are, but
            # this could change), so skipping these datasets seems like the right thing to do.
            pytest.skip("LPG-FTW datasets always create the same tasks, no matter the seed.")

        assert not _equal(setting_1_dict, setting_3_dict)

        # Additionally, explicitly check that either the train schedule or the train envs are
        # different, since the check above could have passed due to some other attribute being
        # different between the two settings.
        if isinstance(setting_1, IncrementalRLSetting) and setting_1.train_envs:
            assert isinstance(setting_3, IncrementalRLSetting)
            # Using custom envs for each task.
            assert not _equal(setting_1.train_envs, setting_3.train_envs)
            assert not _equal(setting_1.val_envs, setting_3.val_envs)
            assert not _equal(setting_1.test_envs, setting_3.test_envs)
        else:
            # Using a single env with a task schedule.
            assert not _equal(setting_1.train_task_schedule, setting_3.train_task_schedule)
            assert not _equal(setting_1.val_task_schedule, setting_3.val_task_schedule)
            assert not _equal(setting_1.test_task_schedule, setting_3.test_task_schedule)

    def test_env_attributes_change(self, setting_kwargs: Dict[str, Any], config: Config):
        """Check that the values of the given attributes do change at each step during
        training.
        """
        setting_kwargs.setdefault("nb_tasks", 2)
        setting_kwargs.setdefault("train_max_steps", 1000)
        setting_kwargs.setdefault("max_episode_steps", 50)
        setting_kwargs.setdefault("test_max_steps", 1000)
        setting = self.Setting(**setting_kwargs)
        assert setting.train_task_schedule

        # NOTE: Have to check for `setting.train_envs` because in that case the task schedule won't
        # be used.
        from sequoia.settings.rl.incremental.setting import IncrementalRLSetting

        if isinstance(setting, IncrementalRLSetting) and setting._using_custom_envs_foreach_task:
            # It would be pretty hard to check for the "task values" in this case, because the
            # custom envs for each task might not be just the same env type but with different
            # attributes!
            pytest.skip("Using custom envs for each task instead of a task schedule.")

        assert all(setting.train_task_schedule.values())
        assert setting.nb_tasks == setting_kwargs["nb_tasks"]
        assert setting.train_steps_per_task == setting_kwargs["train_max_steps"] // setting.nb_tasks
        assert setting.train_max_steps == setting_kwargs["train_max_steps"]

        attributes = set().union(*[task.keys() for task in setting.train_task_schedule.values()])

        method = DummyMethod()

        results = setting.apply(method, config=config)

        assert results
        self.validate_results(setting, method, results)
        # TODO: Need to limit the episodes per step in MonsterKong.
        # In MonsterKong, we might have 0 reward, since this might not even
        # constitute a full episode.
        # assert results.objective

        for attribute in attributes:
            train_values: List[float] = [
                values[attribute]
                for values_dict in method.all_train_values
                for step, values in values_dict.items()
            ]
            assert train_values
            task_schedule_values: List[float] = {
                step: task[attribute] for step, task in setting.train_task_schedule.items()
            }
            self.validate_env_value_changes(
                setting=setting,
                attribute=attribute,
                task_schedule_for_attr=task_schedule_values,
                train_values=train_values,
            )

    def validate_env_value_changes(
        self,
        setting: ContinualRLSetting,
        attribute: str,
        task_schedule_for_attr: Dict[str, float],
        train_values: List[float],
    ):
        """Given an attribute name, and the values of that attribute in the
        task schedule, check that the actual values for that attribute
        encountered during training make sense, based on the type of
        non-stationarity present in this Setting.
        """
        assert len(set(task_schedule_for_attr.values())) == setting.nb_tasks + 1, (
            f"Task schedule should have had {setting.nb_tasks + 1} distinct values for "
            f"attribute {attribute}: {task_schedule_for_attr}"
        )

        if setting.smooth_task_boundaries:
            # Should have one (unique) value for the attribute at each step during training
            # This is the truth condition for the ContinualRLSetting.
            # NOTE: There's an offset by 1 here because of when the env is closed.
            # NOTE: This test won't really work with integer values, but that doesn't matter
            # right now because we don't/won't support changing the values of integer
            # parameters in this "continuous" task setting.
            assert len(set(train_values)) == setting.train_max_steps - 1, (
                f"Should have encountered {setting.train_max_steps-1} distinct values "
                f"for attribute {attribute}: during training!"
            )
        else:
            from ..discrete.setting import DiscreteTaskAgnosticRLSetting

            setting: DiscreteTaskAgnosticRLSetting
            train_tasks = setting.nb_tasks
            unique_attribute_values = set(train_values)

            assert setting.train_task_schedule.keys() == task_schedule_for_attr.keys()
            for k, v in task_schedule_for_attr.items():
                task_dict = setting.train_task_schedule[k]
                assert attribute in task_dict
                assert task_dict[attribute] == v

            assert len(unique_attribute_values) == train_tasks, (
                type(setting),
                attribute,
                unique_attribute_values,
                task_schedule_for_attr,
                setting.nb_tasks,
            )

    def validate_results(
        self,
        setting: ContinualRLSetting,
        method: DummyMethod,
        results: ContinualRLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        assert method.n_task_switches == 0
        assert method.n_fit_calls == 1
        assert not method.received_task_ids
        assert not method.received_while_training

    @pytest.mark.parametrize(
        "batch_size",
        [None, 1, 3],
    )
    @pytest.mark.timeout(60)
    def test_check_iterate_and_step(
        self,
        setting_kwargs: Dict[str, Any],
        batch_size: Optional[int],
    ):
        """Test that the observations are of the right type and shape, regardless
        of wether we iterate on the env by calling 'step' or by using it as a
        DataLoader.
        """
        setting_kwargs.setdefault("num_workers", 0)

        dataset: str = setting_kwargs["dataset"]
        from gym.envs.registration import registry

        if dataset in registry.env_specs:
            with gym.make(dataset) as temp_env:
                expected_x_space = temp_env.observation_space
                expected_action_space = temp_env.action_space
        else:
            # NOTE: Not ideal: Have to create a setting just to get the observation space
            temp_setting = self.Setting(**setting_kwargs)
            # NOTE: Using the test dataloader so the task labels space is a Sparse(Discrete(n)) in
            # the worst case, and so all observations (None or integers) are valid samples.
            with temp_setting.test_dataloader() as temp_env:
                # e = temp_env
                # while e.unwrapped is not e:
                #     print(f"Wrapper of type {type(e)} has obs space of {e.observation_space}")
                #     e = e.env
                # print(f"Unwrapped obs space is {e.observation_space}")
                # assert False, temp_env
                expected_x_space = temp_env.observation_space.x
                expected_action_space = temp_env.action_space
            del temp_setting

        setting = self.Setting(**setting_kwargs)

        if batch_size is not None:
            expected_batched_x_space = batch_space(expected_x_space, batch_size)
            expected_batched_action_space = batch_space(setting.action_space, batch_size)
        else:
            expected_batched_x_space = expected_x_space
            expected_batched_action_space = expected_action_space

        assert setting.observation_space.x == expected_x_space
        assert setting.action_space == expected_action_space

        # TODO: This is changing:
        assert setting.train_transforms == []
        # assert setting.train_transforms == [Transforms.to_tensor, Transforms.three_channels]

        def check_env_spaces(env: gym.Env) -> None:
            if env.batch_size is not None:
                # TODO: This might not be totally accurate, for example because the
                # TransformObservation wrapper applied to a VectorEnv doesn't change the
                # single_observation_space, AFAIR.
                assert env.single_observation_space.x == expected_x_space
                assert env.single_action_space == expected_action_space
                assert isinstance(env.observation_space, TypedDictSpace), (
                    env,
                    env.observation_space,
                )
                assert env.observation_space.x == expected_batched_x_space
                assert env.action_space == expected_batched_action_space
            else:
                assert env.observation_space.x == expected_x_space
                assert env.action_space == expected_action_space

        # FIXME: Move this to an instance method on the test class so that subclasses
        # can change stuff in it.
        def check_obs(obs: ContinualRLSetting.Observations) -> None:
            if isinstance(self.Setting, partial):
                # NOTE: This Happens when we sneakily switch out the self.Setting
                # attribute in other tests (for the SettingProxy for example).
                assert isinstance(obs, self.Setting.args[0].Observations)
            else:
                assert isinstance(obs, self.Setting.Observations)
            assert obs.x in expected_batched_x_space
            # In this particular case here, the task labels should be None.
            # FIXME: For InrementalRL, this isn't correct! TestIncrementalRL should
            # therefore have its own version of this function.
            if self.Setting is ContinualRLSetting:
                assert obs.task_labels is None or all(
                    task_label == None for task_label in obs.task_labels
                )

        with setting.train_dataloader(batch_size=batch_size, num_workers=0) as env:
            assert env.batch_size == batch_size
            check_env_spaces(env)

            # BUG: The dataset's observation space has task_labels as a Discrete, but the task
            # labels are None.
            setting: ContinualRLSetting
            if setting.task_labels_at_train_time:
                if batch_size is not None:
                    assert isinstance(env.observation_space.task_labels, spaces.MultiDiscrete)
                else:
                    assert isinstance(env.observation_space.task_labels, spaces.Discrete)
            elif setting.known_task_boundaries_at_train_time:
                assert isinstance(env.observation_space.task_labels, Sparse)

            obs = env.reset()
            # BUG: TODO: The observation space that we use should actually check with
            # isinstance and over the fields that fit in the space. Here there is a bug
            # because the env observations also have a `done` field, while the space
            # doesnt.
            # assert obs in env.observation_space
            assert obs.x in env.observation_space.x  # this works though.

            # BUG: This doesn't currently work: (would need a tuple value rather than an
            # array.
            # assert obs.task_labels in env.observation_space.task_labels
            assert obs.task_labels in env.observation_space.task_labels
            if batch_size:
                assert obs.x[0] in setting.observation_space.x
                assert obs.task_labels[0] in setting.observation_space.task_labels
            else:
                assert obs in setting.observation_space

            reset_obs = env.reset()
            check_obs(reset_obs)

            # BUG: Environment is closed? (batch_size = 3, dataset = 'CartPole-v0')
            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())

        with setting.val_dataloader(batch_size=batch_size, num_workers=0) as env:
            assert env.batch_size == batch_size
            check_env_spaces(env)

            reset_obs = env.reset()
            check_obs(reset_obs)

            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())

        # NOTE: Limitting the batch size at test time to None (i.e. a single env)
        # because of how the Monitor class works atm.
        batch_size = None
        expected_batched_x_space = expected_x_space
        expected_batched_action_space = expected_action_space

        # NOTE: Need to make sure that the 'directory' passed to the Monitor
        # wrapper is a temp dir. Should be the case, but just checking.
        assert setting.config.log_dir != Path("results")

        with setting.test_dataloader(batch_size=batch_size, num_workers=0) as env:
            assert env.batch_size is None
            check_env_spaces(env)

            reset_obs = env.reset()
            check_obs(reset_obs)

            step_obs, *_ = env.step(env.action_space.sample())
            check_obs(step_obs)

            # NOTE: Can't do this here, unless the episode is over, because the Monitor
            # doesn't want us to end an episode early!
            # for iter_obs in take(env, 3):
            #     check_obs(iter_obs)
            #     _ = env.send(env.action_space.sample())

        with setting.test_dataloader(batch_size=batch_size) as env:
            assert not env.is_closed()
            # NOTE: Can't do this here, unless the episode is over, because the Monitor
            # doesn't want us to end an episode early!
            for iter_obs in take(env, 3):
                check_obs(iter_obs)
                _ = env.send(env.action_space.sample())

    @pytest.mark.no_xvfb
    @pytest.mark.timeout(20)
    @pytest.mark.skipif(
        (not Path("temp").exists()),
        reason="Need temp dir for saving the figure this test creates.",
    )
    @mujoco_required
    def test_show_distributions(self, config: Config):
        setting = self.Setting(
            dataset="half_cheetah",
            max_steps=1_000,
            max_episode_steps=100,
            config=config,
        )

        fig, axes = plt.subplots(2, 3)
        name_to_env_fn = {
            "train": setting.train_dataloader,
            "valid": setting.val_dataloader,
            "test": setting.test_dataloader,
        }
        for i, (name, env_fn) in enumerate(name_to_env_fn.items()):
            env = env_fn(batch_size=None, num_workers=None)

            gravities: List[float] = []
            task_labels: List[Optional[int]] = []
            total_steps = 0
            while not env.is_closed():
                obs = env.reset()
                done = False
                steps_in_episode = 0

                while not done:
                    t = obs.task_labels
                    obs, reward, done, info = env.step(env.action_space.sample())
                    total_steps += 1
                    steps_in_episode += 1
                    y = reward.y

                    gravities.append(env.gravity)
                    print(total_steps, env.gravity)
                    if total_steps > 100:
                        assert env.gravity != -9.81

                    task_labels.append(t)

            x = np.arange(len(gravities))
            axes[0, i].plot(x, gravities, label="gravities")
            axes[0, i].legend()
            axes[0, i].set_title(f"{name} gravities")
            axes[0, i].set_xlabel("Step index")
            axes[0, i].set_ylabel("Value")

            # for task_id in task_ids:
            #     y = [t_counter.get(task_id) for t_counter in t_counters]
            #     axes[1, i].plot(x, y, label=f"task_id={task_id}")
            # axes[1, i].legend()
            # axes[1, i].set_title(f"{name} task_id")
            # axes[1, i].set_xlabel("Batch index")
            # axes[1, i].set_ylabel("Count in batch")

        plt.legend()

        Path("temp").mkdir(exist_ok=True)
        fig.set_size_inches((6, 4), forward=False)
        plt.savefig(f"temp/{self.Setting.__name__}.png")
        # plt.waitforbuttonpress(10)
        # plt.show()


# @pytest.mark.xfail(reason="TODO: pl_bolts DQN only accepts string environment names..")
# def test_dqn_on_env(tmp_path: Path):
#     """ TODO: Would be nice if we could have the models work directly on the
#     gym envs..
#     """
#     from pl_bolts.models.rl import DQN
#     from pytorch_lightning import Trainer

#     setting = ContinualRLSetting()
#     env = setting.train_dataloader(batch_size=None)
#     model = DQN(env)
#     trainer = Trainer(fast_dev_run=True, default_root_dir=tmp_path)
#     success = trainer.fit(model)
#     assert success == 1


def test_passing_task_schedule_sets_other_attributes_correctly():
    # TODO: Figure out a way to test that the tasks are switching over time.
    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_max_steps=10_000,
    )
    assert setting.phases == 1
    assert setting.nb_tasks == 2
    # assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        5_000: {"gravity": 10.0},
        10_000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 10_000
    # assert setting.test_steps_per_task == 5_000

    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_max_steps=2000,
        # test_steps_per_task=100,
    )
    assert setting.phases == 1
    # assert setting.nb_tasks == 2
    # assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        1000: {"gravity": 10.0},
        2000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 2000
    # assert setting.test_steps_per_task == 100


def test_fit_and_on_task_switch_calls():
    setting = ContinualRLSetting(
        dataset="CartPole-v0",
        # nb_tasks=5,
        # train_steps_per_task=100,
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


if MUJOCO_INSTALLED:
    from sequoia.settings.rl.envs.mujoco import (
        ContinualHalfCheetahEnv,
        ContinualHalfCheetahV2Env,
        ContinualHalfCheetahV3Env,
        ContinualHopperEnv,
        ContinualHopperV2Env,
        ContinualHopperV3Env,
        ContinualWalker2dEnv,
        ContinualWalker2dV2Env,
        ContinualWalker2dV3Env,
    )

    @mujoco_required
    @pytest.mark.parametrize(
        "dataset, expected_env_type",
        [
            ("half_cheetah", ContinualHalfCheetahEnv),
            ("halfcheetah", ContinualHalfCheetahEnv),
            ("HalfCheetah-v2", ContinualHalfCheetahV2Env),
            ("HalfCheetah-v3", ContinualHalfCheetahV3Env),
            ("ContinualHalfCheetah-v2", ContinualHalfCheetahV2Env),
            ("ContinualHalfCheetah-v3", ContinualHalfCheetahV3Env),
            ("ContinualHopper-v2", ContinualHopperEnv),
            ("hopper", ContinualHopperEnv),
            ("Hopper-v2", ContinualHopperV2Env),
            ("Hopper-v3", ContinualHopperV3Env),
            ("walker2d", ContinualWalker2dV3Env),
            ("Walker2d-v2", ContinualWalker2dV2Env),
            ("Walker2d-v3", ContinualWalker2dV3Env),
            ("ContinualWalker2d-v2", ContinualWalker2dV2Env),
            ("ContinualWalker2d-v3", ContinualWalker2dV3Env),
        ],
    )
    def test_mujoco_env_name_maps_to_continual_variant(
        dataset: str, expected_env_type: Type[gym.Env]
    ):
        setting = ContinualRLSetting(dataset=dataset, train_max_steps=10_000, test_max_steps=10_000)
        train_env = setting.train_dataloader()
        assert isinstance(train_env.unwrapped, expected_env_type)
