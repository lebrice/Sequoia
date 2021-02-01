import sys
# This "hack" is required so we can run `python examples/continual_rl_demo.py`
sys.path.extend([".", ".."])
from sequoia.methods.stable_baselines3_methods import DQNMethod, A2CMethod
from sequoia.settings import (IncrementalRLSetting, ContinualRLSetting, RLSetting,
                      TaskIncrementalRLSetting)
from sequoia.common.config import Config

if __name__ == "__main__":
   
    # In MountainCar-v0, you can change the gravity and force parameters
    # of the environment.  I change them here for different schedule-based "tasks."  
    # Eventually, we are going to determine if we can transfer the "learning" 
	# between different simulated gravity environments.
    task_schedule = {
        0:      {"gravity": 0.0025, "force": 0.001}, # earth
        1000:   {"gravity": 0.00042, "force": 0.001}, # moon
        2000:   {"gravity": 0.009, "force": 0.001}, # mars
    }

    # Here, we override the default configuration to support interactive
    # render of the simulation. Yes, I like to watch!
    cfg = Config()
    cfg.render = True

    # Let's create a ContinualRLSetting for which to apply methods late
    # on in this app0 for the MountainCar-v0 OpenAI gym environment.
    setting = ContinualRLSetting(
        dataset="MountainCar-v0",
        observe_state_directly=True,
        max_steps=2000,
        train_task_schedule=task_schedule
    )

    # Create the method to use here:
    # NOTE: The DQN method doesn't seem to work nearly as well as A2C.
    # method = DQNMethod(train_steps_per_task=1_000)
    method = A2CMethod(train_steps_per_task=1_000)
    # You could change the hyper-parameters of the method too:
    # method.hparams.buffer_size = 100

    results = setting.apply(method,config=cfg)
    print(results.summary())
