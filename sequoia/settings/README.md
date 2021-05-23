# Sequoia - Settings

### (WIP) Adding a new Setting:

Prerequisites:


- Take a quick look at the `dataclasses` example
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the Settings.




<!-- MAKETREE -->




## Available Settings:


- ## [Setting](base/setting.py)

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


	- ## [IncrementalSetting](assumptions/incremental.py)

		Mixin that defines methods that are common to all 'incremental'
		settings, where the data is separated into tasks, and where you may not
		always get the task labels.

		Concretely, this holds the train and test loops that are common to the
		ClassIncrementalSetting (highest node on the Passive side) and ContinualRL
		(highest node on the Active side), therefore this setting, while abstract,
		is quite important. 



	- ## [SLSetting](passive/passive_setting.py)

		Setting where actions have no influence on future observations. 

		For example, supervised learning is a Passive setting, since predicting a
		label has no effect on the reward you're given (the label) or on the next
		samples you observe.


		- ## [ClassIncrementalSetting](passive/cl/class_incremental_setting.py)

			Supervised Setting where the data is a sequence of 'tasks'.

			This class is basically is the supervised version of an Incremental Setting


			The current task can be set at the `current_task_id` attribute.


			- ## [TaskIncrementalSetting](passive/cl/task_incremental/task_incremental_setting.py)

				Setting where data arrives in a series of Tasks, and where the task
				labels are always available (both train and test time).


				- ## [IIDSetting](passive/cl/task_incremental/iid/iid_setting.py)

					Your 'usual' learning Setting, where the samples are i.i.d.

					Implemented as a variant of Task-Incremental CL, but with only one task.



	- ## [RLSetting](active/active_setting.py)

		LightningDataModule for an 'active' setting.

		This is to be the parent of settings like RL or maybe Active Learning.


		- ## [ContinualRLSetting](active/continual/continual_rl_setting.py)

			Reinforcement Learning Setting where the environment changes over time.

			This is an Active setting which uses gym environments as sources of data.
			These environments' attributes could change over time following a task
			schedule. An example of this could be that the gravity increases over time
			in cartpole, making the task progressively harder as the agent interacts with
			the environment.


			- ## [IncrementalRLSetting](active/continual/incremental/incremental_rl_setting.py)

				Continual RL setting the data is divided into 'tasks' with clear boundaries.

				By default, the task labels are given at train time, but not at test time.

				TODO: Decide how to implement the train procedure, if we give a single
				dataloader, we might need to call the agent's `on_task_switch` when we reach
				the task boundary.. Or, we could produce one dataloader per task, and then
				implement a custom `fit` procedure in the CLTrainer class, that loops over
				the tasks and calls the `on_task_switch` when needed.


				- ## [TaskIncrementalRLSetting](active/continual/incremental/task_incremental/task_incremental_rl_setting.py)

					Continual RL setting with clear task boundaries and task labels.

					The task labels are given at both train and test time.


					- ## [RLSetting](active/continual/incremental/task_incremental/stationary/iid_rl_setting.py)

						Your usual "Classical" Reinforcement Learning setting.

						Implemented as a TaskIncrementalRLSetting, but with a single task.

