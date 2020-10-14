# <repo name>
Potential names:
- (ResearchTree?) (something tree-related?)


## Installation
Requires python >= 3.7

```console
git submodule init
git submodule update
pip install -r requirements.txt
```

## TODOS:
- [ ] Add more documentation.
- [ ] Make sure Wandb logging works well and doesn't produce garbage.
- [ ] Validate/Test out the RL branch.
- [ ] ImageNet Training on the Mila cluster.
- [ ] Test/debug Multi-GPU training on Mila cluster.
- [ ] Test/debug Multi-node-Multi-GPU training on Mila cluster.
- [ ] Add support for more of the datasets from Continuum
- [ ] Add support for using 'native' iid datasets in the IID setting, or normal gym environments in the 'classical RL' setting. 


## Getting Started:
- Take a look at Pytorch Lightning
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the experiments.
- Take a look at the `Setting` class, which looks a 


### Adding a new Setting or Method:
- [ ] Write a test that demonstrates how your new setting or method should behave.
- [ ] Implement your new method / setting
- [ ] Write some tests for **every single module**. Have them be easy to read so people could ideally understand how things work by simply reading the tests.
- [ ] Finally, graft your new method or setting onto the tree by adding them to `all_methods` or `all_settings`, respectively.

<!-- MAKETREE -->
   



## Available Settings:


- ## [Setting](settings/base/setting.py)

	Base class for all research settings in ML: Root node of the tree. 

	A 'setting' is loosely defined here as a learning problem with a specific
	set of assumptions, restrictions, and an evaluation procedure.

	For example, Reinforcement Learning is a type of Setting in which we assume
	that an Agent is able to observe an environment, take actions upon it, and 
	receive rewards back from the environment. Some of the assumptions include
	that the reward is dependant on the action taken, and that the actions have
	an impact on the environment's state (and on the next observations the agent
	will receive). The evaluation procedure consists in trying to maximize the
	reward obtained from an environment over a given number of steps.

	This 'Setting' class should ideally represent the most general learning
	problem imaginable, with almost no assumptions about the data or evaluation
	procedure.

	This is a dataclass. Its attributes are can also be used as command-line
	arguments using `simple_parsing`.


	- ## [ActiveSetting](settings/active/active_setting.py)

		LightningDataModule for an 'active' setting.

		TODO: Use this for something like RL or Active Learning.


		- ## [ContinualRLSetting](settings/active/rl/continual_rl_setting.py)

			Reinforcement Learning Setting where the environment changes over time.

			This is an Active setting which uses gym environments as sources of data.
			These environments' attributes could change over time following a task
			schedule. An example of this could be that the gravity increases over time
			in cartpole, making the task progressively harder as the agent interacts with
			the environment.


			- ## [ClassIncrementalRLSetting](settings/active/rl/class_incremental_rl_setting.py)

				Continual RL setting the data is divided into 'tasks' with clear boundaries.

				By default, the task labels are given at train time, but not at test time.

				TODO: Decide how to implement the train procedure, if we give a single
				dataloader, we might need to call the agent's `on_task_switch` when we reach
				the task boundary.. Or, we could produce one dataloader per task, and then
				implement a custom `fit` procedure in the CLTrainer class, that loops over
				the tasks and calls the `on_task_switch` when needed.


				- ## [TaskIncrementalRLSetting](settings/active/rl/task_incremental_rl_setting.py)

					Continual RL setting with clear task boundaries and task labels.

					The task labels are given at both train and test time.


					- ## [RLSetting](settings/active/rl/iid_rl_setting.py)

						Your usual "Classical" Reinforcement Learning setting.

						Implemented as a TaskIncrementalRLSetting, but with a single task.


	- ## [PassiveSetting](settings/passive/passive_setting.py)

		Setting where actions have no influence on future observations. 

		For example, supervised learning is a Passive setting, since predicting a
		label has no effect on the reward you're given (the label) or on the next
		samples you observe.


		- ## [ClassIncrementalSetting](settings/passive/cl/class_incremental_setting.py)

			Supervised Setting where the data is a sequence of 'tasks'.

			This class is basically is the supervised version of an Incremental Setting


			The current task can be set at the `current_task_id` attribute.


			- ## [TaskIncrementalSetting](settings/passive/cl/task_incremental/task_incremental_setting.py)

				Setting where data arrives in a series of Tasks, and where the task
				labels are always available (both train and test time).


				- ## [IIDSetting](settings/passive/cl/task_incremental/iid/iid_setting.py)

					Your 'usual' learning Setting, where the samples are i.i.d.

					Implemented as a variant of Task-Incremental CL, but with only one task.





## Registered Methods (so far):

- ## [BaselineMethod](methods/baseline_method.py) 

	 - Target setting: [Setting](settings/base/setting.py)

	Versatile Baseline method which targets all settings.

	Uses pytorch-lightning's Trainer for training and LightningModule as model. 

	Uses a [BaselineModel](methods/models/baseline_model/baseline_model.py), which
	can be used for:
	- Self-Supervised training with modular auxiliary tasks;
	- Semi-Supervised training on partially labeled batches;
	- Multi-Head prediction (e.g. in task-incremental scenario);

- ## [RandomBaselineMethod](methods/random_baseline.py) 

	 - Target setting: [Setting](settings/base/setting.py)

	Baseline method that gives random predictions for any given setting.

	This method doesn't have a model or any parameters. It just returns a random
	action for every observation.


