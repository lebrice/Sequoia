from methods.stable_baselines3_methods import DQNMethod
from settings import (ClassIncrementalRLSetting, ContinualRLSetting, RLSetting,
                      TaskIncrementalRLSetting)


if __name__ == "__main__":
    
    task_schedule = {
        0:      {"gravity": 10, "length": 0.2},
        1000:   {"gravity": 10, "length": 1.2},
        2000:   {"gravity": 10, "length": 0.2},
    }

    setting = ClassIncrementalRLSetting(
        dataset="CartPole-v1",
        observe_state_directly=True,
        train_task_schedule=task_schedule,
        valid_task_schedule=task_schedule,
        test_task_schedule=task_schedule,
    )
    # Create the method to use here:
    method = DQNMethod(train_steps_per_task=1_000)
    
    # We can change the hyper-parameters like so:
    # method.hparams.buffer_size = 100

    results = setting.apply(method)
    
    # def evaluation_procedure(algo) -> Resutls:
    
    
    
    print(results.summary())
