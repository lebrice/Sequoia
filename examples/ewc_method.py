# Imports:
from dataclasses import dataclass
from typing import Dict, Tuple, Type, Optional
import sys

from simple_parsing.helpers.serialization.encoding import encode
from torch.cuda import device
sys.path.extend([".", ".."])
import gym
import torch
import warnings
from gym import spaces 
from settings import Method
from torch import Tensor, nn 
import torch.nn.functional as F
from simple_parsing import choice
from settings import Setting
from settings.base import Environment
from torch.utils.data import TensorDataset, DataLoader

from settings.assumptions.incremental import IncrementalSetting
from settings.passive import PassiveEnvironment, PassiveSetting
from settings.passive.cl.task_incremental import TaskIncrementalSetting
from settings.passive.cl.class_incremental_setting import ClassIncrementalSetting
from settings.passive.cl.objects import (Actions, Observations,
                                         PassiveEnvironment, Results, Rewards)
from settings.active.rl.class_incremental_rl_setting import ClassIncrementalRLSetting
from settings.active import ActiveEnvironment, ActiveSetting
from settings.active.rl.wrappers import RemoveTaskLabelsWrapper, NoTypedObjectsWrapper

from utils.nngeometry.nngeometry.object import (PVector, PMatDiag, PMatBlockDiag)
from torch.nn.functional import softmax
from utils.nngeometry.nngeometry.generator.jacobian import Jacobian 
from utils.nngeometry.nngeometry.layercollection import LayerCollection

from stable_baselines3 import DQN     
from stable_baselines3.dqn import MlpPolicy  
from stable_baselines3.common.type_aliases import GymEnv 
from stable_baselines3.common.evaluation import evaluate_policy

from settings import (ClassIncrementalRLSetting, ContinualRLSetting, RLSetting,
                      TaskIncrementalRLSetting)

#copied from https://github.com/tfjgeorge/nngeometry
def FIM(model,
        loader,
        representation,
        n_output,
        variant='classif_logits',
        device='cpu',
        function=None,
        layer_collection=None):
    """
    Helper that creates a matrix computing the Fisher Information
    Matrix using closed form expressions for the expectation y|x
    as described in (Pascanu and Bengio, 2013)

    Parameters
    ----------
    model : torch.nn.Module
        The model that contains all parameters of the function
    loader : torch.utils.data.DataLoader
        DataLoader for computing expectation over the input space
    representation : class
        The parameter matrix representation that will be used to store
        the matrix
    n_output : int
        Number of outputs of the model
    variants : string 'classif_logits' or 'regression', optional
            (default='classif_logits')
        Variant to use depending on how you interpret your function.
        Possible choices are:
         - 'classif_logits' when using logits for classification
         - 'regression' when using a gaussian regression model
    device : string, optional (default='cpu')
        Target device for the returned matrix
    function : function, optional (default=None)
        An optional function if different from `model(input)`. If
        it is different from None, it will override the device
        parameter.
    layer_collection : layercollection.LayerCollection, optional
            (default=None)
        An optional layer collection 
    """

    if function is None:
        def function(*d):
            return model(d[0].to(device))

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == 'classif_logits':
        def function_fim(*d):          
            log_probs = torch.log_softmax(function(*d), dim=1)
            probs = torch.exp(log_probs).detach()
            return (log_probs * probs**.5)

    elif variant == 'regression':
        def function_fim(*d):
            estimates = model(function(*d))
            return estimates

    elif variant == 'dqn':
        def function_fim(*d):
            log_probs = torch.log_softmax(function(*d), dim=1)
            probs = torch.exp(log_probs).detach()
            return (log_probs * probs**.5)


    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         loader=loader,
                         function=function_fim,
                         n_output=n_output)
    return representation(generator)

class DQN_EWC(DQN):
    def __init__(self, fim_representation, ewc_coefficient:float, total_timesteps_fim:int, *args, **kwargs) -> None:    
        super(DQN_EWC, self).__init__(verbose=1, _init_setup_model=False, env=None, *args, **kwargs)

        ########################################
        ######### EWC specific things ##########
        self.FisherMatrix: Jacobian = None
        self.FIM_representation = fim_representation       
        self.ewc_coefficient: float = ewc_coefficient
        self.last_task_train_env:ActiveEnvironment = None

        self._previous_task_id = 0
        self.previous_model_weights: PVector = None
        self._n_switches: int = 0 
        self.total_timesteps_fim = total_timesteps_fim
        ########################################
    
    def set_env(self, env: GymEnv) -> None:
        self.last_task_train_env=env
        return super().set_env(env)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q = torch.gather(current_q, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)

            ### add EWC regularizer ###
            ewc_reg = self.get_ewc_loss()
            #metrics_dict["ewc_regularizer"] = ewc_reg
            loss += self.ewc_coefficient * ewc_reg
            ###########################
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        # logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # logger.record("train/loss", np.mean(losses))

    def on_task_switch(self, task_id: Optional[int]):
        """ Executed when the task switches (to either a known or unknown task).
        """
        if self.previous_model_weights is None and self._n_switches == 0:
            print("Starting the first task, no EWC update.")
            
        elif task_id is None or task_id != self._previous_task_id:
            # NOTE: We also switch between unknown tasks.
            print(f"Switching tasks: {self._previous_task_id} -> {task_id}: ")
            print(f"Updating the EWC 'anchor' weights.")
                 
            self.previous_model_weights = PVector.from_model(self.policy.q_net).clone().detach() 
            observation_collection = torch.Tensor(self.replay_buffer.observations[:self.total_timesteps_fim]).squeeze().to(self.device)
            dataloader = DataLoader(TensorDataset(observation_collection), batch_size=100)
            self.FIM = FIM(model=self.policy.q_net,      
                        loader=dataloader,                   
                        representation=self.FIM_representation,
                        n_output=self.action_space.n,
                        variant='dqn',
                        device=self.device.type)        
        self._n_switches += 1
        self._previous_task_id = task_id
    
    def get_ewc_loss(self) -> Tensor:
        """Gets an 'ewc-like' regularization loss.
        """
        if self.previous_model_weights is None:
            # We're in the first task: do nothing.
            return 0.
        
        v_current = PVector.from_model(self.q_net)      
        regularizer = self.FIM.vTMv(v_current - self.previous_model_weights)
        return regularizer
    
    def forward(self,observation:Observations):
        return self.q_net(observation)
    
    def get_action(self, observations: Observations, task_id: int = None):
        #TODO: multihead DQN that would make use of task id
        observation = observations[0]
        #task_id = observation[1]
        action, _ = self.predict(observation, deterministic=True)
        return action

class Supervised_EWC(nn.Module):

    def __init__(self,
                 nb_tasks: int, 
                 learning_rate: float,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 fim_representation, 
                 ewc_coefficient: float,
                 max_epochs: int,
                 multihead: bool,
                 device):
        super().__init__()
        image_shape = observation_space[0].shape
        assert image_shape == (3, 28, 28)
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        self.n_classes = action_space.n
        self.nb_tasks = nb_tasks 
        self.learning_rate = learning_rate
        image_channels = image_shape[0]

        self.multihead = multihead        
        self._previous_task_id: Optional[int] = None  
        self._previous_task_id: Optional[int] = None
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n_classes),
        )
        self.classifiers = nn.ModuleList([self.classifier for _ in range(nb_tasks)]) if self.multihead else self.classifier
        self.loss = nn.CrossEntropyLoss()

        self.max_epochs: int = max_epochs
        self.early_stop_patience: int = 2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        ########################################
        ######### EWC specific things ##########
        self.FisherMatrix: Jacobian = None
        self.FIM_representation = fim_representation        
        self.ewc_coefficient = ewc_coefficient
        self.last_task_train_env:PassiveEnvironment = None

        self.device = device
        self.previous_model_weights = None
        self._n_switches: int = 0
        self._previous_task_id: int = 0
        ########################################

    def forward(self, observations: Observations) -> Tensor:
        if self.multihead:
            try:
                x, task_labels = observations
                task_label = torch.unique(task_labels)
                assert len(task_label)==1
            except ValueError:
                # this path should be used when calculating FIM
                x = observations
                task_label = [self._previous_task_id] if self._previous_task_id is not None else [0]
                features = self.encoder(x)
                logits = self.classifiers[task_label[0]](features)
                return torch.log_softmax(logits, dim=1)
            except TypeError as e:
                if not self.training:
                    warnings.warn('Note, EWC requieres task label at training and test time')
                raise e
            features = self.encoder(x)
            logits = self.classifiers[task_label[0]](features)
            return logits
        else:
            x = observations[0]
            features = self.encoder(x)
            logits = self.classifiers(features)
            return logits

        
        
    def training_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def validation_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        # Since we're training on a Passive environment, we get both
        # observations and rewards.
        observations: Observations = batch[0]
        rewards: Rewards = batch[1]
        image_labels = rewards.y

        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)

        loss = self.loss(logits, image_labels)
                
        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        
        return loss, metrics_dict
    
    def on_task_switch(self, new_task_id: Optional[int]):
        if self._previous_task_id is None and self._n_switches == 0:
            print("Starting the first task, no EWC update.")
            
        elif new_task_id is None or new_task_id != self._previous_task_id:
            # NOTE: We also switch between unknown tasks.
            print(f"Switching tasks: {self._previous_task_id} -> {new_task_id}: ")
            print(f"Updating the EWC 'anchor' weights.")                
            
            self.previous_model_weights = PVector.from_model(self).clone().detach()
            dataloader = DataLoader(self.last_task_train_env.dataset,
                                                        batch_size=self.last_task_train_env.batch_size)
            self.FIM = FIM(model=self,      
                        loader=dataloader,                 
                        representation=self.FIM_representation,
                        n_output=self.n_classes,
                        variant='classif_logits',    
                        device=self.device.type)
            self._previous_task_id = new_task_id
        
        self._n_switches += 1
    
    def train_supervised(self, train_env: PassiveEnvironment, valid_env:PassiveEnvironment):
        # configure() will have been called by the setting before we get here.
        
        #we save a reference the training environment of the current task to use it during task_switch of EWC 
        self.last_task_train_env = train_env
        ####################################
        import tqdm  
        from numpy import inf
        self.train()
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):               
            # self.train_supervised(train_env, valid_env)
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.training_step(batch)

                    ### add EWC regularizer ###
                    ewc_reg = self.get_ewc_loss()
                    metrics_dict["ewc_regularizer"] = ewc_reg
                    loss += self.ewc_coefficient * ewc_reg
                    ###########################
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_pbar.set_postfix(**metrics_dict)
            

            # Validation loop:
            self.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.validation_step(batch)
                    epoch_val_loss += batch_val_loss
                    val_pbar.set_postfix(**metrics_dict, val_loss=epoch_val_loss)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = i
            if i - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
    
    def get_ewc_loss(self) -> Tensor:
        """Gets an 'ewc-like' regularization loss.
        """
        if self.previous_model_weights is None:
            # We're in the first task: do nothing.
            return 0.
        
        v_current = PVector.from_model(self)     
        regularizer = self.FIM.vTMv(v_current - self.previous_model_weights)
        return regularizer

    def get_action(self, observations):    
            logits = self(observations)
            y_pred = logits.argmax(dim=-1)
            return y_pred

class EWC(Method, target_setting= IncrementalSetting):  
    """ Minimal example of a Method targetting the Class-Incremental CL setting.
    
    For a quick intro to dataclasses, see examples/dataclasses_example.py    
    """

    @dataclass
    class HParams:     
        """ Hyper-parameters of the demo model. """
        # Learning rate of the optimizer.
        learning_rate: float = 0.001
            
        # Fisher information type   
        FIM_representation: str = choice('diagonal', 'block_diagonal', default='diagonal')
            
        # Coeficient of EWC regularizer
        ewc_coefficient: float = 100   

        #number of timesteps for learning
        total_timesteps_train: int = 5000

        # #number of episodes for evaluation
        # n_eval_episodes: int = 10

        #number of timesteps for FIM calculation
        total_timesteps_fim: int = 10000
        
        # #number of timesteps for demonstration of agents performance
        # timesteps_demo: int = 100

        #how many steps of the model to collect transitions for before learning starts (DQN)
        learning_starts: int = 50000

        #replay buffer size (DQN)
        buffer_size: int = 1000

        #maximum number of epochs (supervised)
        max_epochs: int = 10

        #debug
        debug: bool = False
            
    def __init__(self, hparams: HParams):
        self.hparams: EWC.HParams = hparams

        # We will create those when `configure` will be called, before training.
        self.model = None            

    def configure(self, setting: Setting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        FIM_representation = PMatDiag if self.hparams.FIM_representation=='diagonal' else PMatBlockDiag

        if isinstance(setting, PassiveSetting):
            self.model = Supervised_EWC(
                nb_tasks = setting.nb_tasks,
                learning_rate=self.hparams.learning_rate,
                observation_space=setting.observation_space,
                action_space=setting.action_space,
                reward_space=setting.reward_space,
                fim_representation=FIM_representation,
                ewc_coefficient=self.hparams.ewc_coefficient,
                max_epochs = self.hparams.max_epochs if not self.hparams.debug else 1,
                multihead=True if isinstance(setting,TaskIncrementalSetting) else False,
                device=self.config.device
            )
        else:  
            self.model = DQN_EWC(fim_representation=FIM_representation,
                learning_rate=self.hparams.learning_rate,
                ewc_coefficient=self.hparams.ewc_coefficient,
                learning_starts = self.hparams.learning_starts if not self.hparams.debug else 0,
                total_timesteps_fim=self.hparams.total_timesteps_fim if not self.hparams.debug else 100,
                buffer_size = self.hparams.buffer_size if not self.hparams.debug else 100,
                policy=MlpPolicy,
                device=self.config.device) 
         
    def train_supervised(self, train_env: PassiveEnvironment, valid_env:PassiveEnvironment):
        return self.model.train_supervised(train_env, valid_env)

    def train_rl(self, train_env: ActiveEnvironment, valid_env:ActiveEnvironment):
        self.last_task_train_env = train_env  
        train_env = RemoveTaskLabelsWrapper(train_env)
        train_env = NoTypedObjectsWrapper(train_env)
        valid_env = RemoveTaskLabelsWrapper(valid_env)
        valid_env = NoTypedObjectsWrapper(valid_env)   

        if self.model.observation_space == None:            
            self.model.observation_space = train_env.observation_space
        if self.model.action_space is None:
            self.model.action_space = train_env.action_space
        ####################################
        # set the new environment and learn from it
        self.model.set_env(train_env)
        if self.model.policy is None:
            self.model._setup_model()    
        self.model.learn(total_timesteps=self.hparams.total_timesteps_train if not self.hparams.debug else 100)
        ####################################
        #evaluate 
        # mean_reward, std_reward = evaluate_policy(self.model, valid_env, n_eval_episodes=self.hparams.n_eval_episodes)
        # metrics_dict = {'mean_reward':mean_reward,'std_reward':std_reward}
        ####################################

    def fit(self, train_env: Environment, valid_env: Environment):
        if isinstance(train_env, PassiveEnvironment):
            return self.train_supervised(train_env, valid_env)   
        elif isinstance(train_env, ActiveEnvironment):
            return self.train_rl(train_env, valid_env)

    def on_task_switch(self, task_id: Optional[int]):
        if hasattr(self.model, 'on_task_switch'):
            self.model.on_task_switch(task_id) 
    
    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """ 
        with torch.no_grad():
            action = self.model.get_action(observations)
        print(action)
        print(observations[0])
        return self.target_setting.Actions(action)

def demo():
    from simple_parsing import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(EWC.HParams, dest="hparams")
    args = parser.parse_args()
    hparams: EWC.HParams = args.hparams
    method = EWC(hparams=hparams)
    
    # task_schedule = {
    #     0:      {"gravity": 10, "length": 0.2},
    #     1000:   {"gravity": 100, "length": 1.2},
    #     2000:   {"gravity": 10, "length": 0.2},
    # }
    # setting = TaskIncrementalRLSetting(
    #     dataset="CartPole-v1",
    #     observe_state_directly=True,
    #     max_steps=2000,
    #     train_task_schedule=task_schedule,
    # )
     
    #setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)
    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    setting.apply(method)
if __name__ == "__main__":
    demo()