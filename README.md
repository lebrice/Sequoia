# (placeholder repo name)
Potential names:
- (ResearchTree?) (something tree-related?)


## Installation
Requires python >= 3.7

```console
pip install -r requirements.txt
```

## Getting Started - Adding a new Method:

### Prerequisites:

- Take a quick look at the [demo script](examples/quick_demo.py), which can be run using `python examples/quick_demo.py`.

### Steps:

1. Choose a target setting from the tree (See the "Available Settings" section below).

2. Create a new subclass of [`Method`](settings/base/bases.py), with the chosen target setting.

    Your class should implement the following methods:
    - `fit(train_env, valid_env)`
    - `get_actions(observations, observation_space) -> Actions`
    
    The following methods are optional, but can be very useful to help customize how your method is used at train/test time:
    - `configure(setting: Setting)`
    - `on_task_switch(task_id: Optional[int])`
    - `test(test_env)`

    ```python
    class MyNewMethod(Method, target_setting=ClassIncrementalSetting):
        ... # Your code here.

        def fit(self, train_env: DataLoader, valid_env: DataLoader):
            # Train your model however you want here.
            self.trainer.fit(
                self.model,
                train_dataloader=train_env,
                val_dataloaders=valid_env,
            )
        
        def get_actions(self,
                        observations: Observations,
                        observation_space: gym.Space) -> Actions:
            # Return an "Action" (prediction) for the given observations.
            # Each Setting has its own Observations, Actions and Rewards types,
            # which are based on those of their parents.
            return self.model.predict(observations.x)

        def on_task_switch(self, task_id: Optional[int]):
            #This method gets called if task boundaries are known in the current
            #setting. Furthermore, if task labels are available, task_id will be
            # the index of the new task. If not, task_id will be None.
            # For example, you could do something like this:
            self.model.current_output_head = self.model.output_heads[task_id]
    ```

3. Running / Debugging your method:
 
    (at the bottom of your script, for example)
    ```python
    if __name__ == "__main__":
        ## 1. Create the setting you want to apply your method on.
        # First option: Create the Setting directly in code:
        setting = ClassIncrementalSetting(dataset="cifar10", nb_tasks=5)
        # Second option: Create the Setting from the command-line:
        setting = ClassIncrementalSetting.from_args()
        
        ## 2. Create your Method, however you want.
        my_method = MyNewMethod()

        ## 3. Apply your method on the setting to obtain results.
        results = setting.apply(my_method)
        # Optionally, display the results.
        print(results.summary())
        results.make_plots()
    ```

4. (WIP): Adding your new method to the tree:

    - Place the script/package that defines your Method inside of the `methods` folder.

    - Add the `@register_method` decorator to your Method definition, for example:

        ```python
        from sequoia.methods import register_method

        @register_method
        class MyNewMethod(Method, target_setting=ClassIncrementalSetting):
            name: ClassVar[str] = "my_new_method"
            ...
        ```

    - To launch an experiment using your method, run the following command:

        ```console
        python main.py --setting <some_setting_name> --method my_new_method
        ```
        To customize how your method gets created from the command-line, override the two following class methods:
        - `add_argparse_args(cls, parser: ArgumentParser)`
        - `from_argparse_args(cls, args: Namespace) -> Method`

    - Create a `<your_method_script_name>_test.py` file next to your method script. In it, write unit tests for every module/component used in your Method. Have them be easy to read so people can ideally understand how the components of your Method work by simply reading the tests.

        - (WIP) To run the unittests locally, use the following command: `pytest methods/my_new_method_test.py`

    - Then, write a functional test that demonstrates how your new method should behave, and what kind of results it expects to produce. The easiest way to do this is to implement a `validate_results(setting: Setting, results: Results)` method.
        - (WIP) To debug/run the "integration tests" locally, use the following command: `pytest -x methods/my_new_method_test.py --slow`

    - Create a Pull Request, and you're good to go!


## (WIP): Getting Started - Adding a new Setting:

Prerequisites:

- Take a quick look at the `dataclasses` example
- Take a quick look at [simple_parsing](https://github.com/lebrice/SimpleParsing) (A python package I've created) which we use to generate the command-line arguments for the Settings.




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


