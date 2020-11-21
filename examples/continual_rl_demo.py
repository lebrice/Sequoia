from methods.stable_baselines3_methods import DQNMethod
from settings import (ClassIncrementalRLSetting, ContinualRLSetting, RLSetting,
                      TaskIncrementalRLSetting)


if __name__ == "__main__":
    task_schedule = {
        0:      {"gravity": 10, "length": 0.2},
        1000:   {"gravity": 10, "length": 1.2},
        2000:   {"gravity": 10, "length": 0.2},
    }
    setting = ContinualRLSetting(
        dataset="CartPole-v1",
        observe_state_directly=True,
        max_steps=2000,
        train_task_schedule=task_schedule,
    )
    # Create the method to use here:
    method = DQNMethod(train_steps_per_task=1_000)
    # You could change the hyper-parameters of the method too:
    # method.hparams.buffer_size = 100

    results = setting.apply(method)
    print(results.summary())
