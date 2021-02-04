# Sequoia - The Research Tree 

A Playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.

## Please note: This is still very much a Work-In-Progress!

## Motivation:
Most applied ML research generally either proposes new Settings (research problems), new Methods (solutions to such problems), or both.

- When proposing new Settings, researchers almost always have to reimplement or heavily modify existing solutions before they can be applied onto their new problem.

- Likewise, when creating new Methods, it's often necessary to first re-create the experimental setting of other baseline papers, or even the baseline methods themselves, as experimental conditions may be *slightly* different between papers!

The goal of this repo is to:

- Organize various research Settings into an inheritance hierarchy (a tree!), with more *general*, challenging settings with few assumptions at the top, and more constrained problems at the bottom.

- Provide a mechanism for easily reusing existing solutions (Methods) onto new Settings through **Polymorphism**!

- Allow researchers to easily create new, general Methods and quickly gather results on a multitude of Settings, ranging from Supervised to Reinforcement Learning!


## Installation
Requires python >= 3.7

```console
git clone https://www.github.com/lebrice/Sequoia.git
cd Sequoia
pip install -e .
```

### Additional Installation Steps for Mac

Install the latest XQuartz app from here: https://www.xquartz.org/releases/index.html

Then run the following commands on the terminal:

```console
mkdir /tmp/.X11-unix 
sudo chmod 1777 /tmp/.X11-unix 
sudo chown root /tmp/.X11-unix/
```

## Documentation overview:
- ### **[Getting Started / Examples (read this first!)](examples/)**
- ### Runing Experiments (below)
- ### [Settings overview](sequoia/settings/)
- ### [Methods overview](sequoia/methods/)


### Current Settings & Assumptions:

| Setting | Active/Passive? | clear task boundaries? | task labels at train time | task labels at test time | # of tasks ? |
| -----   | --------------  | ---------------------- | ------------------------- | ------------------------ | ------------ |
| [ContinualRLSetting](sequoia/settings/active/continual/continual_rl_setting.py) | Active | no | no | no | 1* |
| [IncrementalRLSetting](sequoia/settings/active/continual/incremental/incremental_rl_setting.py) | Active | **yes** | **yes** | no | ≥1 |
| [TaskIncrementalRLSetting](sequoia/settings/active/continual/incremental/task_incremental/task_incremental_rl_setting.py) | Active | **yes** | **yes** | **yes** | ≥1 |
| [RLSetting](sequoia/settings/active/continual/incremental/task_incremental/stationary/iid_rl_setting.py) | Active | **yes** | **yes** | **yes** | **1** |
| [ClassIncrementalSetting](sequoia/settings/passive/cl/class_incremental_setting.py) | Passive | **yes** | **yes** | no | ≥1 |
| [TaskIncrementalSetting](sequoia/settings/passive/cl/task_incremental/task_incremental_setting.py) | Passive | **yes** | **yes** | **yes** | ≥1 |
| [IIDSetting](sequoia/settings/passive/cl/task_incremental/iid/iid_setting.py) | Passive | **yes** | **yes** | **yes** | **1** |

#### Notes

- **Active / Passive**:
	Active settings are Settings where the next observation depends on the current action, i.e. where actions influence future observations, e.g. Reinforcement Learning.
	Passive settings are Settings where the current actions don't influence the next observations (e.g. Supervised Learning.)

- **Bold entries** in the table mark constant attributes which cannot be
   changed from their default value.

- \*: The environment is changing constantly over time in `ContinualRLSetting`, so
    there aren't really "tasks" to speak of.



### Running experiments

--> **(Reminder) First, take a look at the [Examples](/examples)** <--

#### Directly in code:

```python
from sequoia.settings import TaskIncrementalSetting
from sequoia.methods import BaselineMethod
setting = TaskIncrementalSetting(dataset="mnist")
method = BaselineMethod(max_epochs=1)

results = setting.apply(method)
print(results.summary())
```

#### From the command-line:
```console
sequoia --setting <some_setting> --method <some_method>  (arguments)
```
For example:
- Run the BaselineMethod on task-incremental MNIST, with one epoch per task, and without wandb:
	```console
	sequoia --setting task_incremental --dataset mnist --method baseline --max_epochs 1 --no_wandb
	```
- Run the PPO Method from stable-baselines3 on an incremental RL setting, with the default dataset (CartPole) and 5 tasks: 
	```console
	sequoia --setting incremental_rl --nb_tasks 5 --method ppo --steps_per_task 10_000
	```

- Running multiple experiments (wip):

	If you leave out the `--method` argument above, the experiment will compare the results of all the methods applicable to the chosen Setting.

	Likewise, if you leave the `--setting` option unset, the experiment will evaluate the performance of the selected method on all its applicable settings (WIP: and a table will be shown).

