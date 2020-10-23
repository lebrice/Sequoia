from typing import Dict, Union
from collections import deque

import gym
import torch
import pl_bolts.models.rl.common.gym_wrappers
from pl_bolts.models.rl import DQN, Reinforce, DoubleDQN, VanillaPolicyGradient
from pl_bolts.models.rl.common.gym_wrappers import (BufferWrapper,
                                                    FireResetEnv,
                                                    ImageToPyTorch,
                                                    MaxAndSkipEnv,
                                                    ProcessFrame84,
                                                    ScaledFloatFrame)
from pytorch_lightning import Trainer
from simple_parsing import ArgumentParser

from common.config import Config, TrainerConfig
from settings import ContinualRLSetting, Method
from settings.active.rl import ContinualRLSetting
from settings.active.rl.wrappers import RemoveTaskLabelsWrapper, NoTypedObjectsWrapper
#  --------- PL BOLTS PATCH ----------
# Changed this here to also accept envs directly rather than just strings.

def make_environment(env_name):
    """Convert environment with wrappers"""
    if isinstance(env_name, str):
        env = gym.make(env_name)
    else:
        env = env_name
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

pl_bolts.models.rl.common.gym_wrappers.make_environment = make_environment
# -------------------

# Change at pl_bolts/models/rl/reinforce_model.py:97
# Change at pl_bolts/models/rl/vanilla_policy_gradient_model.py:91
##
# self.env = gym.make(env) if isinstance(env, str) else env
##

""" Models that work:
- DQN
- DoubleDQN (only for Breakout and envs with action meanings)
- DuelingDQN
- PERDQN
- NoisyDQN

Models that don't work yet:
BUG: pl_bolts/models/rl/common/agents.py:109 
- Reinforce
- VanillaPolicyGradient

"""

# Also: for , Reinforce
# pl_bolts/models/rl/common/agents.py:109 is really dumb.

from pl_bolts.models.rl import DQN, Reinforce, DoubleDQN, VanillaPolicyGradient, DuelingDQN, PERDQN, NoisyDQN

# DQN, DoubleDQN, work atm.
class CustomModel(NoisyDQN):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self._hparams.pop("env")
    
    @classmethod
    def add_model_specific_args(cls, arg_parser: ArgumentParser):
        super().add_model_specific_args(arg_parser)
        # Remove the '--env' argument:
        from argparse import Action
        action: Action
        env_action_index = [i for i, action in enumerate(arg_parser._actions)
                            if "--env" in action.option_strings][0]
        arg_parser._handle_conflict_resolve(None, [("--env", arg_parser._actions[env_action_index])])



class DQNMethod(Method, target_setting=ContinualRLSetting):
    def __init__(self, trainer_config: TrainerConfig, dqn_params: Dict = None):
        super().__init__()
        self.dqn_params = dqn_params or {}
        self.trainer_config = trainer_config or TrainerConfig()
        
        self.model: Optional[CustomModel] = None
        self.trainer: Trainer

    def configure(self, setting: ContinualRLSetting):
        self.trainer = self.trainer_config.make_trainer()
        # Ask the setting not to batch observations during training, since the
        # DQN model needs to add its own hacks on top of the env.
        setting.train_batch_size = None
        setting.test_batch_size = 1
        self.test_batch_size = 1
        self.test_buffer = deque(maxlen=4)

    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        train_env = RemoveTaskLabelsWrapper(train_env)
        train_env = NoTypedObjectsWrapper(train_env)
        valid_env = RemoveTaskLabelsWrapper(valid_env)
        valid_env = NoTypedObjectsWrapper(valid_env)

        if self.model is None:
            self.model = CustomModel(train_env, **self.dqn_params)
        else:
            assert False, f"Should only be called once atm."
            # Not sure how we could 'update' the env, given that the model
            # built a bunch of buffers, etc stuff on top of it..
            self.model.env = train_env
            self.model.test_env = valid_env
        self.trainer.fit(
            self.model,
            # train_dataloader=train_env,
            # val_dataloaders=valid_env,
        )
        # TODO: Don't The model can't handle testing below yet.
        test_results = self.trainer.test(verbose=True)
        print(f"test results: {test_results}")
        print(f"Exiting early because the rest is TODO.")
        exit()
        
    
    def get_actions(self, observations: ContinualRLSetting.Observations, action_space: gym.Space) -> ContinualRLSetting.Actions:
        state = observations.x
        # OK so the DQN model is built to handle a sequence of 4 observations?
        # something like that. So we have to do a bit of a "hack" to get it to
        # work here, where we create a buffer of size 4, and populate it with
        # random guesses at first, and once its filled, we can actually predict.
        # This assumes that we're being asked to give actions for a sequence of
        # observations. 
        
        # Not sure in which order the DQN expects the sequence to be.
        self.test_buffer.append(state)
        if len(self.test_buffer) < 4:
            print(f"Returning random action since we don't yet have 4 observations in the buffer.")
            return action_space.sample()
        
        fake_batches = torch.as_tensor(self.test_buffer)
        assert fake_batches.shape[0] == 4
        assert fake_batches.shape[1] == self.test_batch_size
        fake_batch = fake_batches.reshape((-1, *fake_batches.shape[2:]))

        values = self.model(fake_batch)
        assert False, values.shape
        # return super().fit(train_env=train_env, valid_env=valid_env,)


if __name__ == "__main__":    
    parser = ArgumentParser(add_dest_to_option_strings=False)
    # Add the arguments for configuring the Setting:
    parser.add_arguments(ContinualRLSetting, "setting")
    # parser.add_arguments(Config, "config")

    parser.add_arguments(TrainerConfig, "trainer_config")
    CustomModel.add_model_specific_args(parser)

    args = parser.parse_args()
    setting: ContinualRLSetting = args.setting
    config = None
    # config: Config = args.config
    # setting.config = config
    
    trainer_config: TrainerConfig = args.trainer_config
    
    dqn_params = vars(args)
    dqn_params.pop("setting")
    dqn_params.pop("config", None)
    dqn_params.pop("trainer_config")
    
    method = DQNMethod(trainer_config=trainer_config, dqn_params=dqn_params)    
    results = setting.apply(method, config=config)
    print("Results:")
    print(results.summary())
