from .tasks import make_continuous_task, is_supported
from sequoia.conftest import mujoco_required
import pytest
from typing import Type
from sequoia.settings.rl.envs import (
    MujocoEnv,
    ContinualHalfCheetahV2Env,
    ContinualHalfCheetahV3Env,
    ContinualHopperEnv,
    ContinualWalker2dEnv,
    ContinualHalfCheetahEnv,
)


@mujoco_required
@pytest.mark.parametrize(
    "env_type",
    [
        ContinualHalfCheetahV2Env,
        ContinualHalfCheetahV3Env,
        ContinualHopperEnv,
        ContinualWalker2dEnv,
        ContinualHalfCheetahEnv,
    ],
)
def test_mujoco_tasks(env_type: Type[MujocoEnv]):
    assert is_supported("HalfCheetah-v2")

    from gym.envs.mujoco import HalfCheetahEnv

    # We shouldn't mark the *original* envs as supported, rather, we should only mark
    # our variants as supported.
    assert not is_supported(HalfCheetahEnv)

    assert is_supported(env_type)

    task = make_continuous_task(env_type, step=0, change_steps=[0, 100, 200])
    assert task == {"gravity": -9.81}

    task_a = make_continuous_task(
        env_type, step=100, change_steps=[0, 100, 200], seed=123
    )
    task_b = make_continuous_task(
        env_type, step=100, change_steps=[0, 100, 200], seed=123
    )
    task_c = make_continuous_task(
        env_type, step=100, change_steps=[0, 100, 200], seed=456
    )
    # NOTE: Not sure that this will always give exactly the same result, since idk how
    # seeding is dependant on the machine running the code.
    # assert task == {'gravity': -10.134188877055529}
    assert task_a == task_b
    assert task_a != task_c
