from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.active import ActiveEnvironment
from sequoia.settings import (
    ActiveSetting,
    Observations,
    Rewards,
    Setting,
    SettingType,
    Actions,
    Environment,
)
from sequoia.settings.passive import PassiveEnvironment
from dataclasses import dataclass

# Hparams include all hyperparameters for all methods
from Utils import BaseHParams, seed_everything
from simple_parsing import ArgumentParser
from typing import Dict, Optional, Tuple
from Models import Model, model_types_map, optimizers_map, schedulers_map, ActorCritic
import torch
from torch import Tensor
from numpy import inf, floor
import tqdm
import gym
import wandb
from wandb.wandb_run import Run
from gym import spaces
import math
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from Utils.rl_utils import sample_action
import inspect


class BaseMethod(Method, target_setting=Setting):
    """BaseMethod as a base for both SL and RL Settings
    """    
    @dataclass
    class HParams(BaseHParams):
        pass

    def __init__(self, hparams: BaseHParams = None) -> None:
        """initialization f the base method class

        Args:
            hparams (BaseHParams, optional): Hyperparameters used by the experiment. Defaults to None.
        """        
        self.hparams: BaseHParams = hparams or BaseHParams()
        self.model: Model
        self.optimizer: torch.optim.Optimizer
        self._scheduler_step = False
        self.scheduler = None
        self.buffer = None
        self.n_classes = None
        self.smoothing = 0.04  # for total loss plotting
        self.prev_loss = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._is_rl = False

        if torch.cuda.is_available():
            # cudnn auto tuner for faster inference
            torch.backends.cudnn.benchmark = True
            # reproducibility
            seed_everything(self.hparams.seed)

    def configure(self, setting: SettingType):
        """Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.

        Args:
            setting (SettingType): Current setting from Sequoia (ClassIncremental RL or SL)
        """ 
        # This example is intended for classification / discrete action spaces.
        setting.num_workers = self.hparams.num_workers
        setting.batch_size = self.hparams.batch_size

        observation_space = setting.observation_space
        action_space = setting.action_space
        reward_space = setting.reward_space
        if isinstance(setting, ActiveSetting):
            # Default batch size of 1 in RL
            self.hparams.batch_size = 1
            self.max_rl_train_steps = setting.steps_per_phase
            # to enable rendering of the gym
            self.render = False
            self._is_rl = True
        else:
            # SL setting
            assert isinstance(action_space, spaces.Discrete)
            assert action_space == reward_space
            self.classes_per_task = setting.increment
        self.n_classes = action_space.n

        # update model to actor critic for RL
        model_type = model_types_map[self.hparams.model_type]
        image_space = observation_space.x

        if self._is_rl:
            self.model = ActorCritic(model_type, image_space, action_space.n).to(
                self.device
            )
        else:
            self.model = model_type(
                image_space, self.n_classes, bic=self.hparams.bic).to(self.device)

        optim_type, optim_defaults = self._get_optim_defaults(
            self.hparams.optimizer)
        if "base_optim" in optim_defaults:
            base_optim_type, base_optim_defaults = self._get_optim_defaults(
                optim_defaults["base_optim"]
            )
            base_optim = base_optim_type(
                self.model.parameters(), **base_optim_defaults,
            )
            optim_defaults.pop("base_optim")
            self.optimizer = optim_type(base_optim, **optim_defaults)
        else:
            self.optimizer = optim_type(
                self.model.parameters(), **optim_defaults,)
        if self.hparams.use_scheduler:
            scheduler_type, scheduler_defaults = schedulers_map[
                self.hparams.scheduler_name
            ]
            if "step" in scheduler_defaults:
                self._scheduler_step = scheduler_defaults["step"]
                scheduler_defaults.pop("step")
            self.scheduler = scheduler_type(
                self.optimizer, **scheduler_defaults)

        self.task = 0
        self.epoch = 0
        # smooth loss initialization
        self.prev_loss = None
        self.best_smooth_loss = inf

        self.task_type = "RL" if self._is_rl else "SL"
        self.iteration_name = "step" if self._is_rl else "epoch"

    def setup_wandb(self, run: Run) -> None:
        run.config.update(self.hparams.__dict__)

    def _get_optim_defaults(self, optim_name):
        optim_type, optim_defaults = optimizers_map[optim_name]
        optim_signature = inspect.signature(
            optim_type.__init__).parameters.keys()
        if "weight_decay" in optim_signature:
            if "weight_decay" not in optim_defaults:
                optim_defaults["weight_decay"] = self.hparams.weight_decay
            else:
                self.hparams.weight_decay = optim_defaults["weight_decay"]
        if "lr" in optim_signature:
            if "lr" not in optim_defaults:
                optim_defaults["lr"] = self.hparams.learning_rate
            else:
                self.hparams.learning_rate = optim_defaults["lr"]
        return optim_type, optim_defaults

    def _method_specific_configure(self, setting: ClassIncrementalSetting):
        """Method specific initialization used for vars and settings needed per method

        Args:
            setting (ClassIncrementalSetting): Setting used in the configuration
        """        
        pass

    def fit(
        self,
        train_env: Environment[Observations, Actions, Rewards],
        valid_env: Environment[Observations, Actions, Rewards],
    ):
        """fitting function that is used for both SL and RL

        Args:
            train_env (Environment[Observations, Actions, Rewards]): training environment can be active or passive
            valid_env (Environment[Observations, Actions, Rewards]): validation environment active or passive
        """
        if self._is_rl:
            self.fit_rl(train_env, valid_env)
        else:
            self.fit_sl(train_env, valid_env)

    def fit_rl(self, train_env: ActiveEnvironment, valid_env: ActiveEnvironment):
        """fitting function that is used for both RL

        Args:
            train_env (Environment[Observations, Actions, Rewards]): training environment  
            valid_env (Environment[Observations, Actions, Rewards]): validation environment 
        """
        nb_epochs = self.pre_fit(train_env)
        all_lengths: List[int] = []
        average_lengths: List[float] = []
        all_rewards: List[float] = []
        episode = 0
        total_steps = 0
        nb_steps = nb_epochs * self.max_rl_train_steps
        while not train_env.is_closed() and total_steps < nb_steps:
            episode += 1
            print(f"Starting Episode {episode}")

            log_probs: List[Tensor] = []
            critic_values: List[Tensor] = []
            rewards: List[Tensor] = []
            entropy_term = 0

            observation: ActiveSetting.Observations = train_env.reset()
            # Convert numpy arrays in the observation into Tensors on the right device.
            observation = observation.torch(device=self.device)

            done = False
            episode_steps = 0
            while not done and total_steps < self.max_rl_train_steps:
                episode_steps += 1
                actor_output, critic_val = self.model.get_action_critic(
                    observation.x.float()
                )
                critic_val = critic_val.cpu().detach().numpy()
                action, log_prob, entropy = sample_action(
                    actor_output, return_entropy_log_prob=True
                )

                new_observation: ActiveSetting.Observations
                reward: ActiveSetting.Rewards
                reward, new_observation, done = self.send_actions(
                    train_env, actor_output
                )
                action = ActiveSetting.Actions(
                    y_pred=action.cpu().detach().numpy())

                if self.render:
                    train_env.render()

                new_observation = new_observation.torch(device=self.device)
                total_steps += 1

                reward_value: float = reward.y

                rewards.append(reward_value)
                critic_values.append(critic_val)
                log_probs.append(log_prob)
                entropy_term += entropy

                observation = new_observation
                # TODO update buffer with new observations

            Qval, _ = self.model.get_action_critic(new_observation.x.float())
            Qval = Qval.detach().cpu().numpy()
            all_rewards.append(np.sum(rewards))
            all_lengths.append(episode_steps)
            average_lengths.append(np.mean(all_lengths[-10:]))

            if episode % 10 == 0:
                print(
                    f"step {total_steps}/{nb_steps}, "
                    f"episode: {episode}, "
                    f"reward: {np.sum(rewards)}, "
                    f"total length: {episode_steps}, "
                    f"average length: {average_lengths[-1]} \n"
                )

            if total_steps >= nb_steps:
                print(f"Reached the limit of {nb_steps} steps.")
                break

            # compute Q values
            Q_values = np.zeros_like(critic_values)
            # Use the last value from the critic as the final value estimate.
            q_value = Qval
            for t, reward in reversed(list(enumerate(rewards))):
                q_value = reward + self.hparams.gamma * q_value
                Q_values[t] = q_value

            # update actor critic
            critic_values = torch.as_tensor(
                critic_values, dtype=torch.float, device=self.device
            )
            Q_values = torch.as_tensor(
                Q_values, dtype=torch.float, device=self.device)
            log_probs = torch.stack(log_probs)

            advantage = Q_values - critic_values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = (
                actor_loss
                + critic_loss
                + self.hparams.entropy_term_coefficient * entropy_term
            )
            # TODO use backward function in base_method
            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()
        self.post_fit()

    def fit_sl(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        nb_epochs = self.pre_fit(train_env)
        for epoch in range(nb_epochs):
            self.model.train()
            print(f"Starting {self.iteration_name} {epoch}")
            self.epoch = epoch
            # Training loop:
            stop = self._train_loop(train_env)

            # Validation loop:
            if not (self.hparams.submission) and not (self.hparams.early_stop_train):
                early_stop = self._validation_loop(valid_env)

                if early_stop:
                    print(f"Early stopping at {self.iteration_name} {epoch}. ")
                    break
            elif self.hparams.early_stop_train:
                if stop:
                    print(
                        f"Early stopping during {self.iteration_name} {epoch}. ")
                    break
        self.post_fit()

    def pre_fit(self, train_env):
        """Prefit called before fitting a new task 

        Args:
            train_env  (Environment[Observations, Actions, Rewards]): training environment

        Returns:
            int: number of epochs used for that task
        """        
        if self._is_rl:
            n_batches_per_task = self.max_rl_train_steps
        else:
            n_batches_per_task = len(train_env)
        if self.hparams.use_scheduler:
            if self.hparams.scheduler_name == "lambdalr":
                # used to tune cyclic learning rate
                start_lr = 1e-7
                end_lr = 0.2

                def lambda_fn(x):
                    return math.exp(
                        x
                        * math.log(end_lr / start_lr)
                        / (self.hparams.max_epochs_per_task * n_batches_per_task)
                    )

                self.scheduler.__init__(self.optimizer, lambda_fn)
            elif self.hparams.scheduler_name == "cyclic":
                step_size_up = 4 * n_batches_per_task
                step_size_down = step_size_up
                self.scheduler.total_size = step_size_up + step_size_down
                self.scheduler.step_ratio = step_size_up / self.scheduler.total_size
            self.best_scheduler = self.scheduler.state_dict()

        if not(self._is_rl):
            # early stopping initialization
            if self.hparams.early_stop:
                self.best_val_loss = inf
                self.best_iteration = 0
                self.patience = self.hparams.early_stop_patience
            if self.hparams.reload_best:
                self.best_model = self.model.state_dict()
                self.best_iteration = 0
                self.best_optimizer = self.optimizer.state_dict()
                self.best_buffer = None
            if self.hparams.early_stop_train:
                self.train_patience = self.hparams.early_stop_train_patience
                self.best_smooth_loss = inf
        print(f"task {self.task}")
        self.prev_loss = None
        nb_epochs = int(floor(self.hparams.max_epochs_per_task))
        return nb_epochs

    def post_fit(self):
        """Called after training the current task
        """        
        if self.hparams.wandb_logging and self.hparams.early_stop:
            wandb.run.summary[
                f"best {self.iteration_name} on {self.task_type} task {self.task}"
            ] = self.best_iteration

        if (
            self.hparams.reload_best
            and not (self.hparams.submission)
            and self.best_model is not None
        ):
            # FIXME need to think of  away to enable reload best during a submission
            print(
                f"Loading model from {self.iteration_name} {self.best_iteration}!")
            self.model.load_state_dict(self.best_model)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(self.best_scheduler)
            self.optimizer.load_state_dict(self.best_optimizer)
            # loading buffer corresponding to best epoch
            # make sure that best buffer was not empty before loading (this could happen in the first task)
            # Warning: the earlier the best epoch happens, the less we added examples to the buffer
            if self.best_buffer is not None and "examples" in self.best_buffer:
                print(f"Loading buffer from epoch {self.best_iteration}!")
                self.buffer.load_state_dict(self.best_buffer)
        if self.hparams.wandb_logging:
            self._additional_wandb_logging()

        # TODO self.task == 11 can be used on last task
        if self.hparams.bic and self.task == 11:
            self._fit_bic()
        self.hparams.max_epochs_per_task *= self.hparams.max_epochs_decay_per_task
        self.hparams.early_stop_patience *= self.hparams.early_stop_decay_per_task
        self.model.eval()

    def _fit_bic(self):
        """BIC based fit
        """        
        print("Starting bic fit")
        self.model.new_task()
        self.model.eval()
        memory_dict = self.buffer.get_data(self.hparams.buffer_size)
        memory_inputs = memory_dict["examples"]
        memory_targets = memory_dict["labels"]
        buffer_data = torch.utils.data.TensorDataset(
            memory_inputs, memory_targets)
        buffer_dataloader = torch.utils.data.DataLoader(
            buffer_data, batch_size=self.hparams.batch_size)
        self.bic_optimizer = torch.optim.Adam([self.model.bic_params], lr=0.1)
        # self.bic_optimizer = torch.optim.Adam(
        #     self.model.bic_layer.parameters(), lr=0.02)
        for l in range(self.hparams.bic_epochs):
            if self.hparams.submission:
                # disable progress bar for faster train time
                bic_pbar = buffer_dataloader
            else:
                postfix = {}
                bic_pbar = tqdm.tqdm(buffer_dataloader)
                bic_pbar.set_description(f"BIC Epoch {l}")
            for i, (inputs, labels) in enumerate(bic_pbar):
                self.bic_optimizer.zero_grad()
                with torch.no_grad():
                    logits = self.model(inputs, bic=False)
                unbiased_logits = self.model.apply_bic(logits)
                loss_bic = self.loss(unbiased_logits, labels)
                loss_bic.backward()
                self.bic_optimizer.step()
                if not (self.hparams.submission):
                    postfix.update({'CrossEntropyLoss': loss_bic.item()})
                    bic_pbar.set_postfix(postfix)
                if self.hparams.wandb_logging:
                    # bic_param = self.model.bic_params.detach().cpu().numpy()
                    wandb.log({f"BIC/Task{self.task}": loss_bic.item(),
                               #            f"BIC/Task{self.task}_w": bic_param[1],
                               #            f"BIC/Task{self.task}_b": bic_param[0]
                               }
                              )

    def _train_loop(self, train_env):
        """Training loop for all batches inside a training environment

        Args:
            train_env  (Environment[Observations, Actions, Rewards]): training environment

        Returns:
            bool: Flag used to stop the training used with early stopping
        """
        stop = False
        torch.set_grad_enabled(True)
        if self.hparams.submission:
            # disable progress bar for faster train time
            train_pbar = train_env
        else:
            postfix = {}
            train_pbar = tqdm.tqdm(train_env)
            train_pbar.set_description(f"Training Epoch {self.epoch}")
        for i, batch in enumerate(train_pbar):
            self.optimizer.zero_grad()
            loss_dict, metrics_dict = self.shared_step(
                batch, environment=train_env, validation=False,
            )
            self._backward_loss(loss_dict)
            self.optimizer.step()
            if not (self.hparams.submission):
                postfix.update(metrics_dict)
                train_pbar.set_postfix(postfix)

            if self.hparams.wandb_logging:
                # FIXME disable wandb for submission
                # FIXME disable metrics_dict population
                wandb_metrics_dict = {}
                for key, value in metrics_dict.items():
                    wandb_metrics_dict[f"Loss/Task{self.task}/"+key] = value
                wandb.log(wandb_metrics_dict)
            if self._scheduler_step:
                self._call_scheduler()
            if self.hparams.early_stop_train:
                if self.train_patience < 0:
                    stop = True
                    break
        if not (self._scheduler_step):
            self._call_scheduler()
        return stop

    def _call_scheduler(self):
        """Calls scheduler if it is enabled
        """        
        if self.scheduler is not None:
            self.scheduler.step()
            if self.hparams.wandb_logging:
                wandb.log(
                    {
                        "Train/Task {}/learning_rate".format(
                            self.task
                        ): self.optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )

    def _validation_loop(self, valid_env):
        """Validation loop after training the training

        Args:
            valid_env  (Environment[Observations, Actions, Rewards]): validation environment

        Returns:
            flag: used to early stop or not
        """        
        self.model.eval()
        torch.set_grad_enabled(False)
        with tqdm.tqdm(valid_env) as val_pbar:
            postfix = {}
            val_pbar.set_description(f"Validation Epoch {self.epoch}")
            epoch_val_loss = 0.0

            for i, batch in enumerate(val_pbar):
                val_loss_dict, metrics_dict = self.shared_step(
                    batch, environment=valid_env, validation=True,
                )
                epoch_val_loss += val_loss_dict["CrossEntropyLoss"]
                postfix.update(metrics_dict, val_loss=epoch_val_loss.item())
                val_pbar.set_postfix(postfix)
            self.model.train()
            torch.set_grad_enabled(True)
            if self.hparams.wandb_logging:
                wandb.log({f"Loss/Task{self.task}/val_loss": epoch_val_loss})
            if self.hparams.early_stop:
                if epoch_val_loss <= self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.best_iteration = self.epoch
                    self.patience = self.hparams.early_stop_patience
                    if self.hparams.reload_best:
                        self.best_model = self.model.state_dict()
                        if self.scheduler is not None:
                            self.best_scheduler = self.scheduler.state_dict()
                        if self.buffer is not None:
                            self.best_buffer = self.buffer.state_dict()
                        self.best_optimizer = self.optimizer.state_dict()
                else:
                    self.patience -= 1
                print(f"Patience is {self.patience}")
                return self.patience < 0
            else:
                return False

    def _create_metric_dict(self, loss_dict, y_pred, image_labels):
        """Creates metric dictionary automatically

        Args:
            loss_dict (dict): includes all losses
            y_pred (tensor): predictions
            image_labels (tensor): image labels

        Returns:
            dict: including metric summary incuding accuracy
        """        
        if self.hparams.submission:
            return {}
        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy.cpu().item()}
        for loss_name, loss_val in loss_dict.items():
            metrics_dict[loss_name] = loss_val.detach().cpu().item()
        return metrics_dict

    def _backward_loss(self, loss_dict, retain_graph=False):
        """back-propagation using input loss dictionary

        Args:
            loss_dict (dict): dictionary of losses used
            retain_graph (bool, optional): flag used to retain graph if we will call backprop twice on same network. Defaults to False.
        """        
        # first step to do a backward on incoming loss using autograd
        n_losses = len(loss_dict)
        loss = 0
        compute_smooth_loss = (
            self.hparams.early_stop_train or self.hparams.wandb_logging
        )
        for loss_indx, loss_name in enumerate(loss_dict):
            loss_val = loss_dict[loss_name]
            loss_val.backward(retain_graph=loss_indx <
                              n_losses - 1 or retain_graph)
            if compute_smooth_loss and loss_name == "dark":
                loss += loss_val.item()
        if compute_smooth_loss:
            if self.prev_loss is not None:
                loss = self.smoothing * loss + \
                    (1 - self.smoothing) * self.prev_loss
            self.prev_loss = loss
            if self.hparams.early_stop_train and self.epoch > 0 and self.task > 0:
                if loss < self.best_smooth_loss:
                    self.best_smooth_loss = loss
                    self.train_patience = self.hparams.early_stop_train_patience
                else:
                    self.train_patience -= 1
            if self.hparams.wandb_logging:
                wandb.log(
                    {"Loss/Task{}/smoothDarkLoss".format(self.task): loss, }
                )

    def _add_to_buffer(self, examples, logits, labels, task_labels, loss_scores):
        """Add data to buffer used for replay based methods

        Args:
            examples (tensor): batch of examples
            logits (tensor): batch of predicted logits
            labels (tensor): batch of labels
            task_labels (list): list of task labels
            loss_scores (tensor): individual losses used to prioritize sampling
        """        
        save_to_buffer = self.buffer is not None
        if self.hparams.save_from_epoch > 0:
            save_to_buffer = (
                save_to_buffer and self.hparams.save_from_epoch <= self.epoch
            )
        if save_to_buffer:
            with torch.no_grad():
                self.buffer.add_data(
                    {
                        "examples": examples,
                        "logits": logits,
                        "labels": labels,
                        "task_labels": task_labels,
                        "loss_scores": loss_scores,
                    }
                )

    def _additional_wandb_logging(self):
        """Logging some extra information to wandb
        """        
        if self.hparams.wandb_logging_buffer and self.buffer is not None:
            import plotly.express as px
            import pandas as pd

            data = self.buffer.get_all_data()
            data.pop("examples")
            df = pd.DataFrame(data)
            # px.colors.cyclical.IceFire
            fig = px.histogram(df, x="labels")
            log_key = "buffer-{}/Task{}".format("labels", self.task)
            wandb.log({log_key: fig})
            log_key = "buffer-{}/Task{}".format("loss_scores", self.task)
            fig = px.histogram(df, x="loss_scores")
            wandb.log({log_key: fig})
        else:
            pass

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            if self._is_rl:
                logits = self.model(torch.from_numpy(observations.x).float())
            else:
                logits = self.model(observations.x)
        if self._is_rl:
            y_pred = sample_action(logits).squeeze()
        else:
            # Get the predicted classes
            y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def send_actions(self, environment: Environment, logits):
        """Send actions to environment works for both SL and RL

        Args:
            environment (Environment): environment used for training
            logits (tensor): predicted logits

        Returns:
            Reward: reward returned from environment
        """        
        reward = None
        done = False
        new_observation = None
        if self._is_rl:
            # FIXME logits might need to be transformed to softmax probs.
            action = sample_action(logits)
            action = ActiveSetting.Actions(
                y_pred=action.cpu().detach().numpy().squeeze()
            )
            new_observation, reward, done, _ = environment.step(action)
        else:
            y_pred = logits.argmax(-1).detach()
            reward = environment.send(Actions(y_pred))
        return reward, new_observation, done

    def compute_base_loss(self, x, environment: Environment, rewards: Rewards):
        """Base crossentropy loss used during training

        Args:
            x (tensor): input batch
            environment (Environment): environment used for loss computation
            rewards (Rewards): rewards returned from environment

        Returns:
            tuple: loss_dict and metric summary
        """        
        loss_dict = {}
        metrics_dict = {}
        if self._is_rl:
            pass
        else:
            logits = self.model(x)
            # getr rewards from env
            if rewards is None:
                rewards, _, _ = self.send_actions(environment, logits)
            assert rewards is not None
            targets = rewards.y.to(self.device)
            loss_dict["crossentropy"] = self.loss(logits, targets)

            metrics_dict = self._create_metric_dict(
                loss_dict, logits.argmax(-1).detach(), targets
            )
        return loss_dict, metrics_dict

    def loss(self, preds, target):
        """Individual cross entropy loss for SL

        Args:
            preds (tensor): predictions from model
            target (tensor): true labels

        Returns:
            tensor: individual losses
        """        
        loss = None
        if self._is_rl:
            pass
        else:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(preds, target)
        return loss

    @ classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @ classmethod
    def from_argparse_args(cls, args):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: BaseHParams = args.hparams
        return cls(hparams=hparams)

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        environment: Environment,
        validation: bool = False,
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.

        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.

            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.

        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).

        validation : bool
            A flag to denote if this shared step is a validation

        Returns
        -------
        Tuple[Dict, Dict]
            dict of losses name and tensor value, and a dict of metrics to be logged.
        """
        raise NotImplementedError("Method should be overriden to work!")

    def on_task_switch(self, task_id: int) -> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        self.task = task_id