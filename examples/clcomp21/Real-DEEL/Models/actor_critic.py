"""wrapper for actor critic that takes model name 
as input and creates a new network based on the input architecture 
with actor as normal forward and critic with a special function
"""
# TODO add auxiliary tasks to the model code
from .model import Model
from gym.spaces import Box
import torch.nn as nn
from .model import Model


class ActorCritic(Model):
    def __init__(self, model_object: Model, image_space: Box, n_classes: int) -> None:
        actor_model = model_object(image_space, n_classes)
        critic_model = model_object(image_space, 1)
        encoder = actor_model.get_encoder()
        actor_decoder = actor_model.get_decoder()
        super(ActorCritic, self).__init__(encoder + actor_decoder, len(encoder))
        self.critic_decoder = nn.Sequential(*critic_model.get_decoder())
        

    def get_action_critic(self, x):
        """takes observation as input and returns actor/ critic

        Args:
            x (tensor): input tensor

        Returns:
            tuple: actor output, critic output
        """        
        x = x.to(self.dummy_param.device)
        encoder_output, actor_output = self.get_penultimate(x)
        critic_output = self.critic_decoder(encoder_output)
        return actor_output, critic_output

