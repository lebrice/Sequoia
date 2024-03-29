import pytest

from sequoia.conftest import monsterkong_required
from sequoia.settings.rl.envs import MetaMonsterKongEnv

from .tasks import make_discrete_task


@monsterkong_required
def test_monsterkong_tasks():
    # assert make_discrete_task.is_supported(MetaMonsterKongEnv)
    task = make_discrete_task(MetaMonsterKongEnv, step=0, change_steps=[0, 100, 200])
    assert task == {"level": 0}

    task = make_discrete_task(MetaMonsterKongEnv, step=100, change_steps=[0, 100, 200])
    assert task == {"level": 1}

    with pytest.raises(RuntimeError):
        _ = make_discrete_task(MetaMonsterKongEnv, step=123, change_steps=[0, 100, 200])
