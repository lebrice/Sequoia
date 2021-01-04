import sys
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from numpy import inf
import tqdm
import gym

from examples.quick_demo import DemoMethod

from sequoia.settings import Method
from sequoia.settings.passive.cl import ClassIncrementalSetting
from sequoia.settings.passive.cl.objects import (Actions, PassiveEnvironment)
from sequoia.settings.passive.cl.objects import Observations, Rewards

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class HatNet(torch.nn.Module):
    """
    @inproceedings{serra2018overcoming,
      title={Overcoming Catastrophic Forgetting with Hard Attention to the Task},
      author={Serra, Joan and Suris, Didac and Miron, Marius and Karatzoglou, Alexandros},
      booktitle={International Conference on Machine Learning},
      pages={4548--4557},
      year={2018}
    }
    """
    def __init__(self, observation_space, taskcla):
        super(HatNet,self).__init__()

        ncha,size,_ = observation_space
        self.taskcla = taskcla

        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),128)
        self.ec3=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),2048)
        self.efc2=torch.nn.Embedding(len(self.taskcla),2048)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""
        self.loss = torch.nn.CrossEntropyLoss()
        self.current_task = 0

    def forward(self,observations,s=50):
        x = observations.x
        t = observations.task_labels
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gfc1,gfc2=masks
        # Gated
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.unsqueeze(2).unsqueeze(3)#.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=h*gc2.unsqueeze(2).unsqueeze(3)#.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h*gc3.unsqueeze(2).unsqueeze(3)#.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
    
        y = None
        task_mask = {}
        for e in set(t.tolist()):
            task_mask[e] = ( t == e )

            if y is None:
                y = self.last[e](h.clone())
            else:
                y[( t == e )] = self.last[e](h.clone())[( t == e )]

        return y, masks

    def mask(self,t,s):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        # Since we're training on a Passive environment, we get both
        # observations and rewards.
        observations: Observations = batch[0]
        rewards: Rewards = batch[1]
        image_labels = rewards.y

        # Get the predictions:
        logits,_ = self(observations)
        y_pred = logits.argmax(-1)

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict

class HatMethod(Method, target_setting=ClassIncrementalSetting):
    """ Minimal example of a Method targetting the Class-Incremental CL setting.
    
    For a quick intro to dataclasses, see examples/dataclasses_example.py    
    """

    @dataclass
    class HParams:
        """ Hyper-parameters of the demo model. """
        # Learning rate of the optimizer.
        learning_rate: float = 0.001
        
        @classmethod
        def from_args(cls) -> "HParams":
            """ Get the hparams of the method from the command-line. """
            from simple_parsing import ArgumentParser
            parser = ArgumentParser(description=cls.__doc__)
            parser.add_arguments(cls, dest="hparams")
            args, _ = parser.parse_known_args()
            return args.hparams

    def __init__(self, hparams: HParams = None):
        self.hparams: HatMethod.HParams = hparams or self.HParams.from_args()
        self.max_epochs: int = 2
        
        # We will create those when `configure` will be called, before training.
        self.model: HatNet
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        # assert False, setting
        setting.batch_size = 128
        taskcla = [ [i+1,int(len(setting.class_order)/setting.nb_tasks)] for i in range(setting.nb_tasks)  ]
        #assert False, taskcla
        self.model = HatNet(
            observation_space = setting.observation_space[0].shape, #Change this
            taskcla = taskcla #Change this
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(batch)
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = i

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """ 
        with torch.no_grad():
            logits, _ = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id: Optional[int]):
        # This method gets called if task boundaries are known in the current
        # setting. Furthermore, if task labels are available, task_id will be
        # the index of the new task. If not, task_id will be None.
        # For example, you could do something like this:
        self.model.current_task = task_id


if __name__ == "__main__":
    # Example: Evaluate a Method on a single CL setting:
    from sequoia.settings import TaskIncrementalSetting
    # from sequoia.settings import TaskIncrementalRLSetting

    ## 1. Creating the setting:
    # First option: create the setting manually:
    setting = TaskIncrementalSetting(dataset="fashionmnist", nb_tasks=5)
    # setting = TaskIncrementalRLSetting(dataset="cartpole", nb_tasks=5)
    # Second option: create the setting from the command-line:
    # setting = TaskIncrementalSetting.from_args()
    
    ## 2. Creating the Method
    method = HatMethod()
    
    ## 3. Applying the method to the setting:
    results = setting.apply(method)
    
    print(results.summary())
    print(f"objective: {results.objective}")
    
    exit()
    
    # Optionally, second part of the demo: Compare the performances of the two
    # methods on all their applicable settings:

    base_method = DemoMethod()
    from examples.demo_utils import compare_results, demo_all_settings
    
    base_results = demo_all_settings(base_method, datasets=["mnist", "fashionmnist"])
    
    new_method = HatMethod()
    improved_results = demo_all_settings(new_method, datasets=["mnist", "fashionmnist"])

    compare_results({
        DemoMethod: base_results,
        ImprovedDemoMethod: improved_results,
    })