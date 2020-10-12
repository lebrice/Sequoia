from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import torch
from pytorch_lightning.core.decorators import auto_move_data
from torch import Tensor, nn

from common.config import Config
from common.layers import Lambda
from common.loss import Loss
from settings import RLSetting
from utils import prod
from utils.logging_utils import get_logger

from .agent import Agent
from .output_heads import OutputHead, ActorCriticHead

logger = get_logger(__file__)

class ActorCritic(Agent):
    @dataclass
    class HParams(Agent.HParams):
        """ HyperParameters of the Actor-Critic Agent. """
        # TODO: Add the hyper-parameters from actor-critic here.

    def __init__(self, setting: RLSetting, hparams: "ActorCritic.HParams", config: Config):
        super().__init__(setting, hparams, config)
        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = torch.dist

    # @auto_move_data
    def forward(self, batch: Union[Tensor, Tuple[Tensor, Tensor, List[bool], Dict]]) -> Dict[str, Tensor]:
        # TODO: What about methods that want to compare the current 'state' and
        # the next 'state'? How would we pass the 'previous state' to it?
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            state, done, info = batch
        else:
            state = batch

        x = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        h_x = self.encode(x)
        output_task_forward_pass = self.output_task(h_x)
        action = output_task_forward_pass["action"]
        predicted_reward = output_task_forward_pass["predicted_reward"]
        return dict(
            x=x,
            h_x=h_x,
            action=action,
            predicted_reward=predicted_reward,
        )

    def create_output_head(self) -> OutputHead:
        """ Create the output head for the task. """
        # TODO: Should the value and policy be different output heads?
        return ActorCriticHead(self.hidden_size, self.output_shape)

    def get_value(self, observation: Tensor, action: Tensor) -> Tensor:
        # FIXME: This is here just for debugging purposes.  
        # assert False, (observation.shape, observation.dtype)
        assert observation.shape[0] == action.shape[0], (observation.shape, action.shape)
        return self.critic([observation, action])

        return torch.rand(self.reward_shape, requires_grad=True)
    
    def get_action(self, observation: Tensor) -> Tensor:
        # FIXME: This is here just for debugging purposes.
        # assert False, (observation.shape, observation.dtype)
        actions = self.setting.val_env.random_actions()
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        return actions

        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        return self.actor(observation)

    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None, loss_name: str = "") -> Loss:
        # TODO: What about methods that want to compare the current 'state' and
        # the next 'state'? How would we pass the 'previous state' to it?
        reward = y
        action = forward_pass["action"]
        predicted_reward = forward_pass["predicted_reward"]
        total_loss: Loss = Loss(loss_name)  
        
        nce = self.loss_fn(predicted_reward, reward.type_as(predicted_reward))
        critic_loss = Loss("critic", loss=nce)
        total_loss += critic_loss
        # TODO: doing this from memory, actor-critic is definitely not that simple.
        actor_loss = Loss("actor", loss=-predicted_reward.mean())
        total_loss += actor_loss

        return total_loss
