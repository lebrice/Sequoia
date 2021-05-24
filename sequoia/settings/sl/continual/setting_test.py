from .setting import ContinualSLSetting
from sequoia.methods import RandomBaselineMethod

def test_debug_setting():
    setting = ContinualSLSetting(dataset="mnist")
    method = RandomBaselineMethod()
    results = setting.apply(method)
    assert False, results.objective


def test_concat_smooth_boundaries():
    from continuum.datasets import MNIST
    from continuum.scenarios import ClassIncremental
    dataset = MNIST("my/data/path", download=True, train=True)
    scenario = ClassIncremental(dataset, increment=2,)

    print(f"Number of classes: {scenario.nb_classes}.")
    print(f"Number of tasks: {scenario.nb_tasks}.")

    from continuum import TaskSet
    from typing import Tuple


    train_datasets = []
    valid_datasets = []
    for task_id, train_taskset in enumerate(scenario):
        train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
        train_datasets.append(train_taskset)
        valid_datasets.append(val_taskset)

    # train_datasets = [Subset(task_dataset, np.arange(20)) for task_dataset in train_datasets]
    train_dataset = smooth_task_boundaries_concat(train_datasets, seed=123)

    from torch.utils.data import DataLoader

    xs = np.arange(len(train_dataset))
    ys = []
    ts = []
    for x, y, t in DataLoader(train_dataset, batch_size=1, shuffle=False):
        print(t.item(), end="")
        ys.append(y)
        ts.append(t)

    import matplotlib.pyplot as plt

    plt.scatter(xs, ys, label="ys")
    plt.scatter(xs, ts, label="ts")
    plt.show()

from collections import Counter

def test_shared_action_space():
    setting = ContinualSLSetting(dataset="mnist", shared_action_space=True)
    c = Counter()
    train_env = setting.train_dataloader(batch_size=128, num_workers=4)
    for _, rewards in train_env:
        if rewards is None:
            rewards = train_env.send(train_env.action_space.sample())

        y = rewards.y.tolist()
        c.update(y)

    assert False, c
