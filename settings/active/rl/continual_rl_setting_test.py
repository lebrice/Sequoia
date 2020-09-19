import gym
import pytest

from .continual_rl_setting import ContinualRLSetting


def test_setting_shapes_match_env():
    setting = ContinualRLSetting(transforms=[])
    assert setting.obs_shape == (400, 600, 3)
    setting = ContinualRLSetting()
    assert setting.obs_shape == (3, 400, 600)


def test_setting_observe_state_directly():
    setting = ContinualRLSetting(observe_state_directly=True)
    assert setting.obs_shape == (4,)


def test_setting_train_dataloader_shapes():
    setting = ContinualRLSetting(dataset="CartPole-v0")
    assert setting.obs_shape == (3, 400, 600)
    setting.prepare_data()
    setting.setup("train")
    dataloader = setting.train_dataloader(batch_size=5)
    for i, batch in enumerate(dataloader):
        assert batch.shape == (5, 3, 400, 600)
        dataloader.send(dataloader.random_actions())
        if i > 10:
            break
    dataloader.close()


@pytest.mark.xfail(reason=f"TODO: DQN model only accepts string environment names...")
def test_dqn_on_env():
    """ TODO: Would be nice if we could have the models work directly on the
    gym envs..
    """
    from pl_bolts.models.rl import DQN
    from pytorch_lightning import Trainer
    setting = ContinualRLSetting(observe_state_directly=False)
    env = setting.train_dataloader(batch_size=5)
    model = DQN("PongNoFrameskip-v4")
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
    assert False
