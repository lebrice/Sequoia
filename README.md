# Sequoia - The Research Tree 

A Playground for research at the intersection of Continual, Reinforcement, and Self-Supervised Learning.

- 5 minute intro: https://www.youtube.com/watch?v=0u48vr96zRQ
- Paper link: https://arxiv.org/abs/2108.01005
- [Continual Supervised Learning Study](https://wandb.ai/sequoia/csl_study) (~6K runs)
- [Continual Reinforcement Learning Study](https://wandb.ai/sequoia/crl_study) (~2300 runs)


## Please note: This is still very much a Work-In-Progress!

If you have any questions or comments, please make an issue!

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


### Basic installation:

```console
$ git clone https://www.github.com/lebrice/Sequoia.git
$ pip install -e Sequoia
```

### Optional Addons
You can also install optional "addons" for Sequoia, each of which either adds new Methods, new environments/datasets, or both.
using either the usual `extras_require` feature of setuptools, or by pip-installing other repositories which register Methods for Sequoia using an `entry_point` in their `setup.py` file.


```console
pip install -e Sequoia[<all>|<plugin name>]
```

### Supervised Learning Plugins

- `avalanche`:
  
  Continual Supervised Learning methods, provided by the [Avalanche](https://github.com/ContinualAI/avalanche) library:
  
    ```console
    $ pip install -e Sequoia[avalanche]
    ```

- `CN-DPM`: Continual Neural Dirichlet Process Mixture model:
    ```console
    $ cd Sequoia
    $ git submodule init  # to setup the submodules
    $ pip install -e sequoia/methods/cn_dpm    
    ```


- `orion`:
  
    Hyper-parameter optimization using [Orion](https://github.com/epistimio/orion)
    ```console
    $ pip install -e Sequoia[orion]
    ```

- `metaworld`:
  
    Continual / Multi-Task Reinforcement Learning environments, thanks to the [metaworld](https://github.com/rlworkgroup/metaworld) package. The usual setup for mujoco needs to be done, Sequoia unfortunately can't do it for you ;(
    ```console
    $ pip install -e Sequoia[metaworld]
    ```

- `monsterkong`:
  
    Continual Reinforcement Learning environment from [the Meta-MonsterKong repo](https://github.com/lebrice/MetaMonsterkong).
    ```console
    $ pip install -e Sequoia[monsterkong]
    ```


- `continual_world`: The Continual World benchmark for Continual Reinforcement learning.
    ```console
    $ cd Sequoia
    $ git submodule init  # to setup the submodules
    $ pip install -e sequoia/methods/continual_world   
    ```

See the `setup.py` file for all the optional extras.

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

| Setting                                                                    | RL vs SL                                                                 | clear task boundaries? | Task boundaries given? | Task labels at training time? | task labels at test time | Stationary context? | Fixed action space |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------- | ---------------------- | ----------------------------- | ------------------------ | ------------------- | ------------------ |
| [Continual RL](sequoia/settings/rl/continual/setting.py)                   | RL                                                                       | no                     | no                     | no                            | no                       | no                  | no(?)              |
| [Discrete Task-Agnostic RL](sequoia/settings/rl/discrete/setting.py)       | RL                                                                       | **yes**                | **yes**                | no                            | no                       | no                  | no(?)              |
| [Incremental RL](sequoia/settings/rl/incremental/setting.py)               | RL                                                                       | **yes**                | **yes**                | **yes**                       | no                       | no                  | no(?)              |
| [Task-Incremental RL](sequoia/settings/rl/task_incremental/setting.py)     | RL                                                                       | **yes**                | **yes**                | **yes**                       | **yes**                  | no                  | no(?)              |
| [Traditional RL](sequoia/settings/rl/task_incremental/setting.py)          | RL                                                                       | **yes**                | **yes**                | **yes**                       | no                       | **yes**             | no(?)              |
| [Multi-Task RL](sequoia/settings/rl/task_incremental/setting.py)           | RL                                                                       | **yes**                | **yes**                | **yes**                       | **yes**                  | **yes**             | no(?)              |
| [Continual SL](sequoia/settings/sl/continual/setting.py)                   | SL                                                                       | no                     | no                     | no                            | no                       | no                  | no                 |
| [Discrete Task-Agnostic SL](sequoia/settings/sl/discrete/setting.py)       | SL                                                                       | **yes**                | no                     | no                            | no                       | no                  | no                 |
| [(Class) Incremental SL](sequoia/settings/sl/incremental/setting.py)       | SL                                                                       | **yes**                | **yes**                | no                            | no                       | no                  | no                 |
| [Domain-Incremental SL](sequoia/settings/sl/domain_incremental/setting.py) | SL                                                                       | **yes**                | **yes**                | **yes**                       | no                       | no                  | **yes**            |
| [Task-Incremental SL](sequoia/settings/sl/task_incremental/setting.py)     | SL                                                                       | **yes**                | **yes**                | **yes**                       | **yes**                  | no                  | no                 |
| [Traditional SL](sequoia/settings/sl/traditional/setting.py)               | SL                                                                       | **yes**                | **yes**                | **yes**                       | no                       | **yes**             | no                 |
| [Multi-Task SL](sequoia/settings/sl/multi_task/setting.py)                 | SL                                                                       | **yes**                | **yes**                | **yes**                       | **yes**                  | **yes**             | no                 |
| <!--                                                                       | [Class-Incremental SL](sequoia/settings/sl/class_incremental/setting.py) | SL                     | **yes**                | **yes**                       | no                       | no                  | no                 | --> |

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
from sequoia.settings import TaskIncrementalSLSetting
from sequoia.methods import BaseMethod
setting = TaskIncrementalSLSetting(dataset="mnist")
method = BaseMethod(max_epochs=1)

results = setting.apply(method)
print(results.summary())
```

#### From the command-line:
```console
sequoia --setting <some_setting> --method <some_method>  (arguments)
```
For example:
- Run the BaseMethod on task-incremental MNIST, with one epoch per task, and without wandb:
    ```console
    sequoia --setting task_incremental_sl --dataset mnist --method base --max_epochs 1
    ```
- Run the PPO Method from stable-baselines3 on an incremental RL setting, with the default dataset (CartPole) and 5 tasks: 
    ```console
    sequoia --setting incremental_rl --nb_tasks 5 --method ppo --steps_per_task 10_000
    ```

More questions? Please let us know by creating an issue!
