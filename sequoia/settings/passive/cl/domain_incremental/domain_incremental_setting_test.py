import itertools

from .domain_incremental_setting import DomainIncrementalSetting
from sequoia.common.spaces import NamedTupleSpace, Image
from gym.spaces import Discrete
import numpy as np


def test_domain_incremental_mnist_setup():
    setting = DomainIncrementalSetting(
        dataset="mnist",
        increment=2,
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    assert setting.observation_space == NamedTupleSpace(x=Image(0.0, 1.0, (3, 28, 28), np.float32), task_labels=Discrete(5))
    assert False, setting.action_space
    for i in range(setting.nb_tasks):
        setting.current_task_id = i
        batch_size = 5
        train_loader = setting.train_dataloader(batch_size=batch_size)
        
        # Find out which classes are supposed to be within this task.
        
        for j, (observations, rewards) in enumerate(itertools.islice(train_loader, 100)):
            x = observations.x
            t = observations.task_labels
            y = rewards.y
            print(i, j, y, t)
            assert all(0 <= y < setting.n_classes_per_task)
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)
            
            reward = train_loader.send([4 for _ in range(batch_size)])
            assert reward is None

        train_loader.close()