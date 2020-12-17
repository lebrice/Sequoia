import sys
# This "hack" is required so we can run `python examples/continual_rl_demo.py`
sys.path.extend([".", ".."])
from sequoia.methods.stable_baselines3_methods import DQNMethod, A2CMethod
from sequoia.settings import (IncrementalRLSetting, ContinualRLSetting, RLSetting,
                      TaskIncrementalRLSetting)


if __name__ == "__main__":
    task_schedule = {
        0:      {"gravity": 10, "length": 0.2},
        1000:   {"gravity": 100, "length": 1.2},
        2000:   {"gravity": 10, "length": 0.2},
    }
    setting = ContinualRLSetting(
        dataset="CartPole-v1",
        observe_state_directly=True,
        max_steps=2000,
        train_task_schedule=task_schedule,
    )
    # Create the method to use here:
    # NOTE: The DQN method doesn't seem to work nearly as well as A2C.
    # method = DQNMethod(train_steps_per_task=1_000)
    method = A2CMethod(train_steps_per_task=1_000)
    # You could change the hyper-parameters of the method too:
    # method.hparams.buffer_size = 100

    results = setting.apply(method)
    print(results.summary())
