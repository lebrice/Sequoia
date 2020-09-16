from .continual_rl_setting import ContinualRLSetting


import gym


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
        if i > 10:
            break
        dataloader.send(dataloader.random_actions())
    dataloader.close()