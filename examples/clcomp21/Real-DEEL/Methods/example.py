from .base_method import BaseMethod


from typing import Dict, Optional, Tuple
from torch import Tensor, nn

from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from dataclasses import dataclass
from Utils import BaseHParams


class Example(BaseMethod):
    @dataclass
    class HParams(BaseHParams):
        # add method specific params with Baseparams having common params
        pass

    def __init__(self, hparams: HParams= None) -> None:
        super().__init__(hparams=hparams or self.HParams())

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
        Tuple[Tensor, Dict]
            dict of losses name and tensor value, and a dict of metrics to be logged.
        """
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]
        
        loss_dict, metrics_dict = self.compute_base_loss(observations.x, environment, rewards)

        return loss_dict, metrics_dict
