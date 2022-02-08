"""
TODO: Tests for the multi-task SL setting.

- Has only one train/test 'phase'
    - The nb_tasks attribute should still reflect the number of tasks.
- on_task_switch should never be called during training
- (not so sure during testing)
- Task labels should be available for both training and testing.
- Classes shouldn't be relabeled.

"""
import itertools

import numpy as np
import pytest
import torch
from gym.spaces import Discrete

from sequoia.common.spaces import Image, TypedDictSpace
from sequoia.settings import Actions, Environment

from .setting import MultiTaskSLSetting


def check_is_multitask_env(env: Environment, has_rewards: bool):
    # dataloader-style:
    for i, (observations, rewards) in itertools.islice(enumerate(env), 10):
        assert isinstance(observations, MultiTaskSLSetting.Observations)
        task_labels = observations.task_labels.cpu().tolist()
        assert len(set(task_labels)) > 1
        if has_rewards:
            assert isinstance(rewards, MultiTaskSLSetting.Rewards)
            # Check that there is no relabelling happening, by checking that there are
            # more different y's then there are usually classes in each batch.
            assert len(set(rewards.y.cpu().tolist())) > 2
        else:
            assert rewards is None

    # gym-style interaction:
    obs = env.reset()
    assert isinstance(env.observation_space, TypedDictSpace)
    space_shapes = {k: s.shape for k, s in env.observation_space.spaces.items()}
    space_dtypes = {k: s.dtype for k, s in env.observation_space.spaces.items()}
    # assert False, (obs.keys(), obs.numpy().keys())
    assert obs.shapes == space_shapes
    assert obs.numpy().shapes == space_shapes

    assert obs.dtypes == space_dtypes
    x_space = env.observation_space.x
    t_space = env.observation_space.task_labels
    assert obs.x in x_space, (obs.x, x_space)
    assert obs.task_labels in t_space, (obs.task_labels, t_space)
    assert isinstance(obs, env.observation_space.dtype)

    assert obs in env.observation_space
    done = False
    steps = 0
    while not done and steps < 10:
        action = Actions(y_pred=torch.randint(10, [env.batch_size]))
        # BUG: convert_tensors seems to be causing issues again: We shouldn't have
        # to manually convert obs to numpy before checking `obs in obs_space`.
        # TODO: Also not super clean that we can't just do `action in action_space`.
        # assert action.numpy() in env.action_space
        assert action.y_pred.numpy() in env.action_space
        obs, reward, done, info = env.step(action)
        assert obs.numpy() in env.observation_space
        assert reward.y in env.reward_space
        steps += 1
        assert done is False
    assert steps == 10


from sequoia.common.config import Config


def test_multitask_setting(config: Config):
    setting = MultiTaskSLSetting(dataset="mnist", config=config)
    assert setting.phases == 1
    assert setting.nb_tasks == 5
    from sequoia.common.spaces.image import ImageTensorSpace
    from sequoia.common.spaces.tensor_spaces import TensorDiscrete

    assert setting.observation_space == TypedDictSpace(
        x=ImageTensorSpace(0.0, 1.0, (3, 28, 28), np.float32, device=config.device),
        task_labels=TensorDiscrete(5, device=config.device),
        dtype=setting.Observations,
    )
    assert setting.action_space == Discrete(10)
    assert setting.config.device.type == "cuda" if torch.cuda.is_available() else "cpu"

    with setting.train_dataloader(batch_size=32, num_workers=0) as train_env:
        check_is_multitask_env(train_env, has_rewards=True)

    with setting.val_dataloader(batch_size=32, num_workers=0) as val_env:
        check_is_multitask_env(val_env, has_rewards=True)


@pytest.mark.xfail(reason="test environments still operate in a 'sequential tasks' way")
def test_multitask_setting_test_env():
    setting = MultiTaskSLSetting(dataset="mnist")

    assert setting.phases == 1
    assert setting.nb_tasks == 5
    assert setting.observation_space == TypedDictSpace(
        x=Image(0.0, 1.0, (3, 28, 28), np.float32), task_labels=Discrete(5)
    )
    assert setting.action_space == Discrete(10)

    # FIXME: Wait, actually, this test environment, will it be shuffled, or not?
    with setting.test_dataloader(batch_size=32, num_workers=0) as test_env:
        check_is_multitask_env(test_env, has_rewards=False)


from sequoia.settings.assumptions.incremental_test import DummyMethod


def test_on_task_switch_is_called_multi_task():
    setting = MultiTaskSLSetting(
        dataset="mnist",
        nb_tasks=5,
        # train_steps_per_task=100,
        # max_steps=500,
        # test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = DummyMethod()
    results = setting.apply(method)
    assert method.n_task_switches == setting.nb_tasks
    assert method.received_task_ids == list(range(setting.nb_tasks))
    assert method.received_while_training == [False for _ in range(setting.nb_tasks)]
