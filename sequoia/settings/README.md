# Sequoia - Settings

### (WIP) Adding a new Setting:

Prerequisites:


- Take a quick look at the `dataclasses` example
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the Settings.




<!-- MAKETREE -->




## Available Settings:


- ## [Setting](sequoia/settings/base/setting.py)

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

  Abstract (required) methods:
  - **apply** Applies a given Method on this setting to produce Results.
  - **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
  - **setup**  (things to do on every accelerator in distributed mode).
  - **train_dataloader** the training environment/dataloader.
  - **val_dataloader** the val environments/dataloader(s).
  - **test_dataloader** the test environments/dataloader(s).

  "Abstract"-ish (required) class attributes:
  - `Results`: The class of Results that are created when applying a Method on
    this setting.
  - `Observations`: The type of Observations that will be produced  in this
      setting.
  - `Actions`: The type of Actions that are expected from this setting.
  - `Rewards`: The type of Rewards that this setting will (potentially) return
    upon receiving an action from the method.


  - ## [RLSetting](sequoia/settings/rl/setting.py)

    LightningDataModule for an 'active' setting.

    This is to be the parent of settings like RL or maybe Active Learning.


    - ## [ContinualRLSetting](sequoia/settings/rl/continual/setting.py)

      Reinforcement Learning Setting where the environment changes over time.

      This is an Active setting which uses gym environments as sources of data.
      These environments' attributes could change over time following a task
      schedule. An example of this could be that the gravity increases over time
      in cartpole, making the task progressively harder as the agent interacts with
      the environment.


      - ## [DiscreteTaskAgnosticRLSetting](sequoia/settings/rl/discrete/setting.py)

        Continual Reinforcement Learning Setting where there are clear task boundaries,
        but where the task information isn't available.


        - ## [IncrementalRLSetting](sequoia/settings/rl/incremental/setting.py)

          Continual RL setting in which:
          - Changes in the environment's context occur suddenly (same as in Discrete, Task-Agnostic RL)
          - Task boundary information (and task labels) are given at training time
          - Task boundary information is given at test time, but task identity is not.


          - ## [TaskIncrementalRLSetting](sequoia/settings/rl/task_incremental/setting.py)

            Continual RL setting with clear task boundaries and task labels.

            The task labels are given at both train and test time.


            - ## [MultiTaskRLSetting](sequoia/settings/rl/multi_task/setting.py)

              Reinforcement Learning setting where the environment alternates between a set
              of tasks sampled uniformly.

              Implemented as a TaskIncrementalRLSetting, but where the tasks are randomly sampled
              during training.


          - ## [TraditionalRLSetting](sequoia/settings/rl/traditional/setting.py)

            Your usual "Classical" Reinforcement Learning setting.

            Implemented as a MultiTaskRLSetting, but with a single task.


            - ## [MultiTaskRLSetting](sequoia/settings/rl/multi_task/setting.py)

              Reinforcement Learning setting where the environment alternates between a set
              of tasks sampled uniformly.

              Implemented as a TaskIncrementalRLSetting, but where the tasks are randomly sampled
              during training.


  - ## [SLSetting](sequoia/settings/sl/setting.py)

    Supervised Learning Setting.

    Core assuptions:
    - Current actions have no influence on future observations.
    - The environment gives back "dense feedback", (the 'reward' associated with all
      possible actions at each step, rather than a single action)

    For example, supervised learning is a Passive setting, since predicting a
    label has no effect on the reward you're given (the label) or on the next
    samples you observe.


    - ## [ContinualSLSetting](sequoia/settings/sl/continual/setting.py)

      Continuous, Task-Agnostic, Continual Supervised Learning.

      This is *currently* the most "general" Supervised Continual Learning setting in
      Sequoia.

      - Data distribution changes smoothly over time.
      - Smooth transitions between "tasks"
      - No information about task boundaries or task identity (no task IDs)
      - Maximum of one 'epoch' through the environment.


      - ## [DiscreteTaskAgnosticSLSetting](sequoia/settings/sl/discrete/setting.py)

        Continual Supervised Learning Setting where there are clear task boundaries, but
        where the task information isn't available.


        - ## [IncrementalSLSetting](sequoia/settings/sl/incremental/setting.py)

          Supervised Setting where the data is a sequence of 'tasks'.

          This class is basically is the supervised version of an Incremental Setting


          The current task can be set at the `current_task_id` attribute.


          - ## [TaskIncrementalSLSetting](sequoia/settings/sl/task_incremental/setting.py)

            Setting where data arrives in a series of Tasks, and where the task
            labels are always available (both train and test time).


            - ## [MultiTaskSLSetting](sequoia/settings/sl/multi_task/setting.py)

              IID version of the Task-Incremental Setting, where the data is shuffled.

              Can be used to estimate the upper bound performance of Task-Incremental CL Methods.


          - ## [DomainIncrementalSLSetting](sequoia/settings/sl/domain_incremental/setting.py)

            Supervised CL Setting where the input domain shifts incrementally.

            Task labels and task boundaries are given at training time, but not at test-time.
            The crucial difference between the Domain-Incremental and Class-Incremental settings
            is that the action space is smaller in domain-incremental learning, as it is a
            `Discrete(n_classes_per_task)`, rather than the `Discrete(total_classes)` in
            Class-Incremental setting.

            For example: Create a classifier for odd vs even hand-written digits. It first be
            trained on digits 0 and 1, then digits 2 and 3, then digits 4 and 5, etc.
            At evaluation time, it will be evaluated on all digits


          - ## [TraditionalSLSetting](sequoia/settings/sl/traditional/setting.py)

            Your 'usual' supervised learning Setting, where the samples are i.i.d.

            This Setting is slightly different than the others, in that it can be recovered in
            *two* different ways:
            - As a variant of Task-Incremental learning, but where there is only one task;
            - As a variant of Domain-Incremental learning, but where there is only one task.


            - ## [MultiTaskSLSetting](sequoia/settings/sl/multi_task/setting.py)

              IID version of the Task-Incremental Setting, where the data is shuffled.

              Can be used to estimate the upper bound performance of Task-Incremental CL Methods.


